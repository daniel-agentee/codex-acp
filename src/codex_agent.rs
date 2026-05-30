use acp::schema::{
    AgentAuthCapabilities, AgentCapabilities, AuthEnvVar, AuthMethod, AuthMethodAgent,
    AuthMethodEnvVar, AuthMethodId, AuthenticateRequest, AuthenticateResponse, CancelNotification,
    ClientCapabilities, CloseSessionRequest, CloseSessionResponse, Implementation,
    InitializeRequest, InitializeResponse, ListSessionsRequest, ListSessionsResponse,
    LoadSessionRequest, LoadSessionResponse, LogoutCapabilities, LogoutRequest, LogoutResponse,
    McpCapabilities, McpServer, McpServerHttp, McpServerStdio, NewSessionRequest,
    NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse, ProtocolVersion,
    ResumeSessionRequest, ResumeSessionResponse, SessionCapabilities, SessionCloseCapabilities,
    SessionId, SessionInfo, SessionInfoUpdate, SessionListCapabilities, SessionNotification,
    SessionResumeCapabilities, SessionSetTitleCapabilities, SessionUpdate,
    SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, SetSessionModeRequest,
    SetSessionModeResponse, SetSessionTitleRequest, SetSessionTitleResponse,
};
use acp::{Agent, Client, ConnectTo, ConnectionTo, Error};
use agent_client_protocol as acp;
use codex_config::{DEFAULT_MCP_SERVER_ENVIRONMENT_ID, McpServerConfig, McpServerTransportConfig};
use codex_core::{
    NewThread, RolloutRecorder, StateDbHandle, ThreadManager, append_thread_name, config::Config,
    find_thread_path_by_id_str, init_state_db, resolve_installation_id, thread_store_from_config,
};
use codex_exec_server::{EnvironmentManager, ExecServerRuntimePaths};
use codex_extension_api::empty_extension_registry;
use codex_login::{
    CODEX_API_KEY_ENV_VAR, OPENAI_API_KEY_ENV_VAR,
    auth::{AuthManager, CodexAuth, read_codex_api_key_from_env, read_openai_api_key_from_env},
};
use codex_protocol::{
    ThreadId,
    protocol::{InitialHistory, SessionSource},
};
use codex_thread_store::{
    ListThreadsParams, SortDirection as StoreSortDirection, ThreadSortKey as StoreThreadSortKey,
    ThreadStore,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tracing::{debug, info};
use unicode_segmentation::UnicodeSegmentation;

use crate::thread::{Thread, ThreadOptions};

/// The Codex implementation of the ACP Agent.
///
/// This bridges the ACP protocol with the existing codex-rs infrastructure,
/// allowing codex to be used as an ACP agent.
pub struct CodexAgent {
    /// Handle to the current authentication
    auth_manager: Arc<AuthManager>,
    /// Capabilities of the connected client
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
    /// The underlying codex configuration
    config: Config,
    /// Thread manager for handling sessions
    thread_manager: ThreadManager,
    /// Store for listing and updating persisted thread metadata
    thread_store: Arc<dyn ThreadStore>,
    /// SQLite-backed Codex state index, when initialization succeeds
    state_db: Option<StateDbHandle>,
    /// Active sessions mapped by `SessionId`
    sessions: Arc<Mutex<HashMap<SessionId, Arc<Thread>>>>,
    /// Session working directories for filesystem sandboxing
    session_roots: Arc<Mutex<HashMap<SessionId, PathBuf>>>,
}

const SESSION_LIST_PAGE_SIZE: usize = 25;
const SESSION_TITLE_MAX_GRAPHEMES: usize = 120;

impl CodexAgent {
    /// Create a new `CodexAgent` with the given configuration
    pub async fn new(
        config: Config,
        codex_linux_sandbox_exe: Option<PathBuf>,
    ) -> std::io::Result<Self> {
        let auth_manager = AuthManager::shared(
            config.codex_home.to_path_buf(),
            false,
            config.cli_auth_credentials_store_mode,
            Some(config.chatgpt_base_url.clone()),
        )
        .await;

        let client_capabilities: Arc<Mutex<ClientCapabilities>> = Arc::default();
        let session_roots: Arc<Mutex<HashMap<SessionId, PathBuf>>> = Arc::default();
        let state_db = init_state_db(&config).await;
        let local_runtime_paths =
            ExecServerRuntimePaths::new(std::env::current_exe()?, codex_linux_sandbox_exe)?;
        let environment_manager = Arc::new(
            EnvironmentManager::from_codex_home(&config.codex_home, Some(local_runtime_paths))
                .await
                .map_err(std::io::Error::other)?,
        );
        let thread_store = thread_store_from_config(&config, state_db.clone());
        let installation_id = resolve_installation_id(&config.codex_home).await?;
        let thread_manager = ThreadManager::new(
            &config,
            auth_manager.clone(),
            SessionSource::Unknown,
            environment_manager,
            empty_extension_registry(),
            None,
            thread_store.clone(),
            state_db.clone(),
            installation_id,
            None,
        );
        Ok(Self {
            auth_manager,
            client_capabilities,
            config,
            thread_manager,
            thread_store,
            state_db,
            sessions: Arc::default(),
            session_roots,
        })
    }

    /// Build and run the ACP agent, serving requests over the given transport.
    pub async fn serve(
        self: Arc<Self>,
        transport: impl ConnectTo<Agent> + 'static,
    ) -> acp::Result<()> {
        let agent = self;
        Agent
            .builder()
            .name("codex-acp")
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: InitializeRequest, responder, _cx| {
                        responder.respond_with_result(agent.initialize(request).await)
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: AuthenticateRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.authenticate(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: LogoutRequest, responder, cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.logout(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: NewSessionRequest, responder, cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        let session_cx = cx.clone();
                        cx.spawn(async move {
                            responder
                                .respond_with_result(agent.new_session(request, session_cx).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: LoadSessionRequest, responder, cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        let session_cx = cx.clone();
                        cx.spawn(async move {
                            responder
                                .respond_with_result(agent.load_session(request, session_cx).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: ResumeSessionRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        let session_cx = cx.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(
                                agent.resume_session(request, session_cx).await,
                            )
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: ListSessionsRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.list_sessions(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: CloseSessionRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.close_session(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: PromptRequest, responder, cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.prompt(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_notification(
                {
                    let agent = agent.clone();
                    async move |notification: CancelNotification, cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            if let Err(e) = agent.cancel(notification).await {
                                tracing::error!("Error handling cancel: {:?}", e);
                            }
                            Ok(())
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_notification!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: SetSessionModeRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.set_session_mode(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: SetSessionTitleRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        let notification_cx = cx.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(
                                agent
                                    .set_session_title(request, |notification| {
                                        notification_cx.send_notification(notification)
                                    })
                                    .await,
                            )
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: SetSessionConfigOptionRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder
                                .respond_with_result(agent.set_session_config_option(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .connect_to(transport)
            .await
    }

    fn session_id_from_thread_id(thread_id: ThreadId) -> SessionId {
        SessionId::new(thread_id.to_string())
    }

    fn get_thread(&self, session_id: &SessionId) -> Result<Arc<Thread>, Error> {
        self.loaded_thread(session_id)
            .ok_or_else(|| Error::resource_not_found(None))
    }

    fn loaded_thread(&self, session_id: &SessionId) -> Option<Arc<Thread>> {
        self.sessions.lock().unwrap().get(session_id).cloned()
    }

    async fn check_auth(&self) -> Result<(), Error> {
        if self.config.model_provider_id == "openai"
            && self.auth_manager.auth().await.is_none()
            // Check if anything changed on disk since the last reload
            && !self.auth_manager.reload().await
        {
            return Err(Error::auth_required());
        }
        Ok(())
    }

    /// Build a session config from base config, working directory, and MCP servers.
    /// This is shared between `new_session` and `load_session`.
    fn build_session_config(
        &self,
        cwd: &Path,
        mcp_servers: Vec<McpServer>,
    ) -> Result<Config, Error> {
        let mut config = self.config.clone();
        config.cwd = cwd.try_into().map_err(Error::into_internal_error)?;
        let cwd = config.cwd.clone();

        // Propagate any client-provided MCP servers that codex-rs supports.
        let mut new_mcp_servers = config.mcp_servers.get().clone();
        for mcp_server in mcp_servers {
            match mcp_server {
                // Not supported in codex
                McpServer::Sse(..) => {}
                McpServer::Http(McpServerHttp {
                    name, url, headers, ..
                }) => {
                    // Codex does not allow whitespace in MCP server names; replace with underscores.
                    let name = name.replace(|c: char| c.is_whitespace(), "_");
                    new_mcp_servers.insert(
                        name,
                        McpServerConfig {
                            transport: McpServerTransportConfig::StreamableHttp {
                                url,
                                bearer_token_env_var: None,
                                http_headers: if headers.is_empty() {
                                    None
                                } else {
                                    Some(headers.into_iter().map(|h| (h.name, h.value)).collect())
                                },
                                env_http_headers: None,
                            },
                            required: false,
                            enabled: true,
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                            disabled_tools: None,
                            enabled_tools: None,
                            disabled_reason: None,
                            scopes: None,
                            oauth: None,
                            oauth_resource: None,
                            tools: Default::default(),
                            environment_id: DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string(),
                            supports_parallel_tool_calls: false,
                            default_tools_approval_mode: None,
                        },
                    );
                }
                McpServer::Stdio(McpServerStdio {
                    name,
                    command,
                    args,
                    env,
                    ..
                }) => {
                    // Codex does not allow whitespace in MCP server names; replace with underscores.
                    let name = name.replace(|c: char| c.is_whitespace(), "_");
                    new_mcp_servers.insert(
                        name,
                        McpServerConfig {
                            transport: McpServerTransportConfig::Stdio {
                                command: command.display().to_string(),
                                args,
                                env: if env.is_empty() {
                                    None
                                } else {
                                    Some(env.into_iter().map(|env| (env.name, env.value)).collect())
                                },
                                env_vars: vec![],
                                cwd: Some(cwd.to_path_buf()),
                            },
                            required: false,
                            enabled: true,
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                            disabled_tools: None,
                            enabled_tools: None,
                            disabled_reason: None,
                            scopes: None,
                            oauth: None,
                            oauth_resource: None,
                            tools: Default::default(),
                            environment_id: DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string(),
                            supports_parallel_tool_calls: false,
                            default_tools_approval_mode: None,
                        },
                    );
                }
                _ => {}
            }
        }

        config
            .mcp_servers
            .set(new_mcp_servers)
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok(config)
    }
}

impl CodexAgent {
    async fn initialize(&self, request: InitializeRequest) -> Result<InitializeResponse, Error> {
        let InitializeRequest {
            protocol_version,
            client_capabilities,
            client_info: _, // TODO: save and pass into Codex somehow
            ..
        } = request;
        debug!("Received initialize request with protocol version {protocol_version:?}",);
        let protocol_version = ProtocolVersion::V1;

        *self.client_capabilities.lock().unwrap() = client_capabilities;

        let mut agent_capabilities = AgentCapabilities::new()
            .prompt_capabilities(PromptCapabilities::new().embedded_context(true).image(true))
            .mcp_capabilities(McpCapabilities::new().http(true))
            .load_session(true)
            .auth(AgentAuthCapabilities::new().logout(LogoutCapabilities::new()));

        agent_capabilities.session_capabilities = SessionCapabilities::new()
            .close(SessionCloseCapabilities::new())
            .list(SessionListCapabilities::new())
            .resume(SessionResumeCapabilities::new())
            .set_title(SessionSetTitleCapabilities::new());

        let mut auth_methods = vec![
            CodexAuthMethod::ChatGpt.into(),
            CodexAuthMethod::CodexApiKey.into(),
            CodexAuthMethod::OpenAiApiKey.into(),
        ];
        // Until codex device code auth works, we can't use this in remote ssh projects
        if std::env::var("NO_BROWSER").is_ok() {
            auth_methods.remove(0);
        }

        Ok(InitializeResponse::new(protocol_version)
            .agent_capabilities(agent_capabilities)
            .agent_info(Implementation::new("codex-acp", env!("CARGO_PKG_VERSION")).title("Codex"))
            .auth_methods(auth_methods))
    }

    async fn authenticate(
        &self,
        request: AuthenticateRequest,
    ) -> Result<AuthenticateResponse, Error> {
        let auth_method = CodexAuthMethod::try_from(request.method_id)?;

        // Check before starting login flow if already authenticated with the same method
        if let Some(auth) = self.auth_manager.auth().await {
            match (auth, auth_method) {
                (
                    CodexAuth::ApiKey(..),
                    CodexAuthMethod::CodexApiKey | CodexAuthMethod::OpenAiApiKey,
                )
                | (CodexAuth::Chatgpt(..), CodexAuthMethod::ChatGpt) => {
                    return Ok(AuthenticateResponse::new());
                }
                _ => {}
            }
        }

        match auth_method {
            CodexAuthMethod::ChatGpt => {
                // Perform browser/device login via codex-rs, then report success/failure to the client.
                let opts = codex_login::ServerOptions::new(
                    self.config.codex_home.to_path_buf(),
                    codex_login::auth::CLIENT_ID.to_string(),
                    None,
                    self.config.cli_auth_credentials_store_mode,
                );

                let server =
                    codex_login::run_login_server(opts).map_err(Error::into_internal_error)?;

                server
                    .block_until_done()
                    .await
                    .map_err(Error::into_internal_error)?;
            }
            CodexAuthMethod::CodexApiKey => {
                let api_key = read_codex_api_key_from_env().ok_or_else(|| {
                    Error::internal_error().data(format!("{CODEX_API_KEY_ENV_VAR} is not set"))
                })?;
                codex_login::login_with_api_key(
                    &self.config.codex_home,
                    &api_key,
                    self.config.cli_auth_credentials_store_mode,
                )
                .map_err(Error::into_internal_error)?;
            }
            CodexAuthMethod::OpenAiApiKey => {
                let api_key = read_openai_api_key_from_env().ok_or_else(|| {
                    Error::internal_error().data(format!("{OPENAI_API_KEY_ENV_VAR} is not set"))
                })?;
                codex_login::login_with_api_key(
                    &self.config.codex_home,
                    &api_key,
                    self.config.cli_auth_credentials_store_mode,
                )
                .map_err(Error::into_internal_error)?;
            }
        }

        self.auth_manager.reload().await;

        Ok(AuthenticateResponse::new())
    }

    async fn logout(&self, _request: LogoutRequest) -> Result<LogoutResponse, Error> {
        self.auth_manager
            .logout()
            .await
            .map_err(Error::into_internal_error)?;
        Ok(LogoutResponse::new())
    }

    async fn new_session(
        &self,
        request: NewSessionRequest,
        cx: ConnectionTo<Client>,
    ) -> Result<NewSessionResponse, Error> {
        // Check before sending if authentication was successful or not
        self.check_auth().await?;

        let NewSessionRequest {
            cwd, mcp_servers, ..
        } = request;
        info!("Creating new session with cwd: {}", cwd.display());

        let config = self.build_session_config(&cwd, mcp_servers)?;
        let num_mcp_servers = config.mcp_servers.len();

        let NewThread {
            thread_id,
            thread,
            session_configured: _,
        } = Box::pin(self.thread_manager.start_thread(config.clone()))
            .await
            .map_err(|_e| Error::internal_error())?;

        let session_id = Self::session_id_from_thread_id(thread_id);
        // Record the session root for filesystem sandboxing.
        self.session_roots
            .lock()
            .unwrap()
            .insert(session_id.clone(), config.cwd.to_path_buf());
        let thread = Arc::new(Thread::new(
            session_id.clone(),
            thread,
            self.auth_manager.clone(),
            Arc::new(self.thread_manager.get_models_manager()),
            self.client_capabilities.clone(),
            ThreadOptions {
                config: config.clone(),
                state_db: self.state_db.clone(),
            },
            cx,
        ));
        let load = thread.load().await?;

        self.sessions
            .lock()
            .unwrap()
            .insert(session_id.clone(), thread);

        debug!("Created new session with {} MCP servers", num_mcp_servers);

        Ok(NewSessionResponse::new(session_id)
            .modes(load.modes)
            .config_options(load.config_options))
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
        cx: ConnectionTo<Client>,
    ) -> Result<LoadSessionResponse, Error> {
        info!("Loading session: {}", request.session_id);
        // Check before sending if authentication was successful or not
        self.check_auth().await?;

        let LoadSessionRequest {
            session_id,
            cwd,
            mcp_servers,
            ..
        } = request;

        self.restore_session(session_id, cwd, mcp_servers, cx, true)
            .await
    }

    async fn resume_session(
        &self,
        request: ResumeSessionRequest,
        cx: ConnectionTo<Client>,
    ) -> Result<ResumeSessionResponse, Error> {
        info!("Resuming session: {}", request.session_id);
        // Check before sending if authentication was successful or not
        self.check_auth().await?;

        let ResumeSessionRequest {
            session_id,
            cwd,
            mcp_servers,
            ..
        } = request;

        let load = self
            .restore_session(session_id, cwd, mcp_servers, cx, false)
            .await?;

        Ok(ResumeSessionResponse::new()
            .modes(load.modes)
            .config_options(load.config_options))
    }

    async fn restore_session(
        &self,
        session_id: SessionId,
        cwd: PathBuf,
        mcp_servers: Vec<McpServer>,
        cx: ConnectionTo<Client>,
        replay_history: bool,
    ) -> Result<LoadSessionResponse, Error> {
        let rollout_path = find_thread_path_by_id_str(
            &self.config.codex_home,
            session_id.0.as_ref(),
            self.state_db.as_deref(),
        )
        .await
        .map_err(|e| Error::internal_error().data(e.to_string()))?
        .ok_or_else(|| Error::resource_not_found(None))?;

        let rollout_items = if replay_history {
            let history = RolloutRecorder::get_rollout_history(&rollout_path)
                .await
                .map_err(|e| Error::internal_error().data(e.to_string()))?;

            match &history {
                InitialHistory::Resumed(resumed) => resumed.history.clone(),
                InitialHistory::Forked(items) => items.clone(),
                InitialHistory::Cleared | InitialHistory::New => Vec::new(),
            }
        } else {
            Vec::new()
        };

        let config = self.build_session_config(&cwd, mcp_servers)?;

        let NewThread {
            thread_id: _,
            thread,
            session_configured: _,
        } = Box::pin(self.thread_manager.resume_thread_from_rollout(
            config.clone(),
            rollout_path,
            self.auth_manager.clone(),
            None,
        ))
        .await
        .map_err(|e| Error::internal_error().data(e.to_string()))?;

        let thread = Arc::new(Thread::new(
            session_id.clone(),
            thread,
            self.auth_manager.clone(),
            Arc::new(self.thread_manager.get_models_manager()),
            self.client_capabilities.clone(),
            ThreadOptions {
                config: config.clone(),
                state_db: self.state_db.clone(),
            },
            cx,
        ));

        if replay_history {
            thread.replay_history(rollout_items).await?;
        }

        let load = thread.load().await?;

        self.session_roots
            .lock()
            .unwrap()
            .insert(session_id.clone(), config.cwd.to_path_buf());
        self.sessions.lock().unwrap().insert(session_id, thread);

        Ok(LoadSessionResponse::new()
            .modes(load.modes)
            .config_options(load.config_options))
    }

    async fn list_sessions(
        &self,
        request: ListSessionsRequest,
    ) -> Result<ListSessionsResponse, Error> {
        self.check_auth().await?;

        let ListSessionsRequest { cwd, cursor, .. } = request;
        let allowed_sources = [
            SessionSource::Cli,
            SessionSource::VSCode,
            SessionSource::Unknown,
        ];
        let cwd_filter = cwd.clone();

        let page = self
            .thread_store
            .list_threads(ListThreadsParams {
                page_size: SESSION_LIST_PAGE_SIZE,
                cursor,
                sort_key: StoreThreadSortKey::UpdatedAt,
                sort_direction: StoreSortDirection::Desc,
                allowed_sources: allowed_sources.to_vec(),
                model_providers: None,
                cwd_filters: cwd.map(|cwd| vec![cwd]),
                archived: false,
                search_term: None,
                use_state_db_only: false,
            })
            .await
            .map_err(|err| {
                Error::internal_error().data(format!("failed to list sessions: {err}"))
            })?;

        let sessions = page
            .items
            .into_iter()
            .filter(|item| {
                allowed_sources.contains(&item.source)
                    && cwd_filter
                        .as_ref()
                        .is_none_or(|filter_cwd| item.cwd.as_path() == filter_cwd.as_path())
            })
            .map(|item| {
                let title = stored_session_title(item.name.as_deref(), &item.preview);
                let updated_at = item.updated_at.to_rfc3339();

                SessionInfo::new(SessionId::new(item.thread_id.to_string()), item.cwd)
                    .title(title)
                    .updated_at(updated_at)
            })
            .collect::<Vec<_>>();

        Ok(ListSessionsResponse::new(sessions).next_cursor(page.next_cursor))
    }

    async fn close_session(
        &self,
        request: CloseSessionRequest,
    ) -> Result<CloseSessionResponse, Error> {
        self.get_thread(&request.session_id)?.shutdown().await?;
        self.thread_manager
            .remove_thread(
                &ThreadId::from_string(&request.session_id.0)
                    .map_err(Error::into_internal_error)?,
            )
            .await;
        self.sessions.lock().unwrap().remove(&request.session_id);
        self.session_roots
            .lock()
            .unwrap()
            .remove(&request.session_id);

        Ok(CloseSessionResponse::new())
    }
    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, Error> {
        info!("Processing prompt for session: {}", request.session_id);
        // Check before sending if authentication was successful or not
        self.check_auth().await?;

        // Get the session state
        let thread = self.get_thread(&request.session_id)?;
        let stop_reason = thread.prompt(request).await?;

        Ok(PromptResponse::new(stop_reason))
    }

    async fn cancel(&self, args: CancelNotification) -> Result<(), Error> {
        info!("Cancelling operations for session: {}", args.session_id);
        self.get_thread(&args.session_id)?.cancel().await?;
        Ok(())
    }

    async fn set_session_mode(
        &self,
        args: SetSessionModeRequest,
    ) -> Result<SetSessionModeResponse, Error> {
        info!("Setting session mode for session: {}", args.session_id);
        self.get_thread(&args.session_id)?
            .set_mode(args.mode_id)
            .await?;
        Ok(SetSessionModeResponse::default())
    }

    async fn set_session_title(
        &self,
        args: SetSessionTitleRequest,
        notify_client: impl FnOnce(SessionNotification) -> Result<(), Error>,
    ) -> Result<SetSessionTitleResponse, Error> {
        info!("Setting session title for session: {}", args.session_id);
        self.check_auth().await?;

        let session_id = args.session_id;
        let title = args.title;
        let thread_id = ThreadId::from_string(&session_id.0).map_err(Error::into_internal_error)?;
        let loaded_thread = self.loaded_thread(&session_id);
        let mut title_updated_in_state = false;

        if let Some(state_db) = self.state_db.as_deref() {
            match state_db.update_thread_title(thread_id, &title).await {
                Ok(true) => {
                    title_updated_in_state = true;
                }
                Ok(false) => {
                    debug!(
                        "Thread metadata unavailable before title update for session: {}",
                        session_id
                    );
                }
                Err(err) => {
                    return Err(Error::internal_error()
                        .data(format!("failed to update thread title: {err}")));
                }
            }
        }

        if let Some(thread) = loaded_thread {
            let notification_title = notification_title_for_update(&title);
            thread.set_title(title, notification_title).await?;
        } else {
            if !title_updated_in_state {
                let rollout_path = find_thread_path_by_id_str(
                    &self.config.codex_home,
                    session_id.0.as_ref(),
                    self.state_db.as_deref(),
                )
                .await
                .map_err(|err| {
                    Error::internal_error().data(format!(
                        "failed to locate thread before title update: {err}"
                    ))
                })?;

                if rollout_path.is_none() {
                    return Err(Error::resource_not_found(None));
                }

                if let Some(state_db) = self.state_db.as_deref() {
                    match state_db.update_thread_title(thread_id, &title).await {
                        Ok(_) => {}
                        Err(err) => {
                            return Err(Error::internal_error()
                                .data(format!("failed to update thread title: {err}")));
                        }
                    }
                }
            }

            let notification_title = notification_title_for_update(&title);
            append_thread_name(&self.config.codex_home, thread_id, &title)
                .await
                .map_err(|err| {
                    Error::internal_error().data(format!("failed to index thread title: {err}"))
                })?;
            notify_client(SessionNotification::new(
                session_id,
                SessionUpdate::SessionInfoUpdate(
                    SessionInfoUpdate::new().title(notification_title),
                ),
            ))?;
        }

        Ok(SetSessionTitleResponse::default())
    }

    async fn set_session_config_option(
        &self,
        args: SetSessionConfigOptionRequest,
    ) -> Result<SetSessionConfigOptionResponse, Error> {
        info!(
            "Setting session config option for session: {} (config_id: {}, value: {:?})",
            args.session_id, args.config_id.0, args.value
        );

        let thread = self.get_thread(&args.session_id)?;

        thread.set_config_option(args.config_id, args.value).await?;

        let config_options = thread.config_options().await?;

        Ok(SetSessionConfigOptionResponse::new(config_options))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodexAuthMethod {
    ChatGpt,
    CodexApiKey,
    OpenAiApiKey,
}

impl From<CodexAuthMethod> for AuthMethodId {
    fn from(method: CodexAuthMethod) -> Self {
        Self::new(match method {
            CodexAuthMethod::ChatGpt => "chatgpt",
            CodexAuthMethod::CodexApiKey => "codex-api-key",
            CodexAuthMethod::OpenAiApiKey => "openai-api-key",
        })
    }
}

impl From<CodexAuthMethod> for AuthMethod {
    fn from(method: CodexAuthMethod) -> Self {
        match method {
            CodexAuthMethod::ChatGpt => Self::Agent(
                AuthMethodAgent::new(method, "Login with ChatGPT").description(
                    "Use your ChatGPT login with Codex CLI (requires a paid ChatGPT subscription)",
                ),
            ),
            CodexAuthMethod::CodexApiKey => Self::EnvVar(
                AuthMethodEnvVar::new(
                    method,
                    format!("Use {CODEX_API_KEY_ENV_VAR}"),
                    vec![AuthEnvVar::new(CODEX_API_KEY_ENV_VAR)],
                )
                .description(format!(
                    "Requires setting the `{CODEX_API_KEY_ENV_VAR}` environment variable."
                )),
            ),
            CodexAuthMethod::OpenAiApiKey => Self::EnvVar(
                AuthMethodEnvVar::new(
                    method,
                    format!("Use {OPENAI_API_KEY_ENV_VAR}"),
                    vec![AuthEnvVar::new(OPENAI_API_KEY_ENV_VAR)],
                )
                .description(format!(
                    "Requires setting the `{OPENAI_API_KEY_ENV_VAR}` environment variable."
                )),
            ),
        }
    }
}

impl TryFrom<AuthMethodId> for CodexAuthMethod {
    type Error = Error;

    fn try_from(value: AuthMethodId) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "chatgpt" => Ok(CodexAuthMethod::ChatGpt),
            "codex-api-key" => Ok(CodexAuthMethod::CodexApiKey),
            "openai-api-key" => Ok(CodexAuthMethod::OpenAiApiKey),
            _ => Err(Error::invalid_params().data("unsupported authentication method")),
        }
    }
}

fn truncate_graphemes(text: &str, max_graphemes: usize) -> String {
    let mut graphemes = text.grapheme_indices(true);

    if let Some((byte_index, _)) = graphemes.nth(max_graphemes) {
        if max_graphemes >= 3 {
            let mut truncate_graphemes = text.grapheme_indices(true);
            if let Some((truncate_byte_index, _)) = truncate_graphemes.nth(max_graphemes - 3) {
                let truncated = &text[..truncate_byte_index];
                format!("{truncated}...")
            } else {
                text.to_string()
            }
        } else {
            let truncated = &text[..byte_index];
            truncated.to_string()
        }
    } else {
        text.to_string()
    }
}

fn format_session_title(message: &str) -> Option<String> {
    let normalized = message.replace(['\r', '\n'], " ");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(truncate_graphemes(trimmed, SESSION_TITLE_MAX_GRAPHEMES))
    }
}

fn stored_session_title(name: Option<&str>, preview: &str) -> Option<String> {
    match name {
        Some(name) if name.trim().is_empty() => None,
        Some(name) => format_session_title(name),
        None => format_session_title(preview),
    }
}

fn notification_title_for_update(title: &str) -> Option<String> {
    format_session_title(title)
}

#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        path::{Path, PathBuf},
        sync::{Arc, Mutex},
    };

    use acp::schema::MaybeUndefined;
    use codex_core::config::ConfigOverrides;
    use codex_core::find_thread_name_by_id;
    use uuid::Uuid;

    use super::*;

    #[test]
    fn stored_session_title_prefers_thread_name() {
        assert_eq!(
            stored_session_title(Some("renamed"), "preview"),
            Some("renamed".to_string())
        );
    }

    #[test]
    fn stored_session_title_falls_back_to_preview_when_name_is_absent() {
        assert_eq!(
            stored_session_title(None, "preview"),
            Some("preview".to_string())
        );
    }

    #[test]
    fn stored_session_title_preserves_explicitly_cleared_name() {
        assert_eq!(stored_session_title(Some("  "), "preview"), None);
    }

    #[test]
    fn notification_title_for_update_matches_session_list_formatting() {
        assert_eq!(
            notification_title_for_update("  custom\ntitle  "),
            Some("custom title".to_string())
        );
        assert_eq!(notification_title_for_update("   "), None);

        let long_title = "a".repeat(SESSION_TITLE_MAX_GRAPHEMES + 1);
        let expected = format!("{}...", "a".repeat(SESSION_TITLE_MAX_GRAPHEMES - 3));
        assert_eq!(notification_title_for_update(&long_title), Some(expected));
    }

    fn write_minimal_rollout_with_id(
        codex_home: &Path,
        id: Uuid,
        user_message: Option<&str>,
    ) -> anyhow::Result<PathBuf> {
        let sessions = codex_home.join("sessions/2024/01/01");
        std::fs::create_dir_all(&sessions)?;

        let file = sessions.join(format!("rollout-2024-01-01T00-00-00-{id}.jsonl"));
        let mut writer = std::fs::File::create(&file)?;
        writeln!(
            writer,
            "{}",
            serde_json::json!({
                "timestamp": "2024-01-01T00:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "id": id.to_string(),
                    "timestamp": "2024-01-01T00:00:00Z",
                    "cwd": ".",
                    "originator": "test",
                    "cli_version": "test",
                    "model_provider": "test-provider"
                }
            })
        )?;

        if let Some(message) = user_message {
            writeln!(
                writer,
                "{}",
                serde_json::json!({
                    "timestamp": "2024-01-01T00:00:00.000Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": message,
                        "kind": "plain"
                    }
                })
            )?;
        }

        Ok(file)
    }

    #[tokio::test]
    async fn set_session_title_persists_unloaded_session() -> anyhow::Result<()> {
        let codex_home =
            std::env::temp_dir().join(format!("codex-acp-unloaded-title-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&codex_home)?;

        let mut config = Config::load_with_cli_overrides_and_harness_overrides(
            vec![],
            ConfigOverrides::default(),
        )
        .await?;
        config.codex_home = codex_home.clone().try_into()?;
        config.model_provider_id = "test-provider".to_string();

        let agent = CodexAgent::new(config, None).await?;
        let thread_uuid = Uuid::new_v4();
        let thread_id = ThreadId::from_string(&thread_uuid.to_string())?;
        write_minimal_rollout_with_id(&codex_home, thread_uuid, None)?;
        let session_id = SessionId::new(thread_id.to_string());
        let notifications = Arc::new(Mutex::new(Vec::new()));
        let captured_notifications = notifications.clone();

        agent
            .set_session_title(
                SetSessionTitleRequest::new(session_id.clone(), "Renamed historical session"),
                |notification| {
                    captured_notifications.lock().unwrap().push(notification);
                    Ok(())
                },
            )
            .await?;

        assert_eq!(
            codex_core::find_thread_name_by_id(&agent.config.codex_home, &thread_id).await?,
            Some("Renamed historical session".to_string())
        );

        let notifications = notifications.lock().unwrap();
        assert!(
            notifications.iter().any(|notification| {
                notification.session_id == session_id
                    && matches!(
                        &notification.update,
                        SessionUpdate::SessionInfoUpdate(update)
                            if matches!(
                                &update.title,
                                MaybeUndefined::Value(title) if title == "Renamed historical session"
                            )
                    )
            }),
            "expected SessionInfoUpdate for unloaded rename; got: {notifications:#?}"
        );
        drop(notifications);

        std::fs::remove_dir_all(codex_home).ok();

        Ok(())
    }

    #[tokio::test]
    async fn set_session_title_persists_db_backed_unloaded_session_to_name_index()
    -> anyhow::Result<()> {
        let codex_home =
            std::env::temp_dir().join(format!("codex-acp-db-backed-title-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&codex_home)?;

        let mut config = Config::load_with_cli_overrides_and_harness_overrides(
            vec![],
            ConfigOverrides::default(),
        )
        .await?;
        config.codex_home = codex_home.clone().try_into()?;
        config.model_provider_id = "test-provider".to_string();

        let agent = CodexAgent::new(config, None).await?;
        let thread_uuid = Uuid::new_v4();
        let thread_id = ThreadId::from_string(&thread_uuid.to_string())?;
        let rollout_path = write_minimal_rollout_with_id(&codex_home, thread_uuid, None)?;
        let session_id = SessionId::new(thread_id.to_string());

        agent
            .set_session_title(
                SetSessionTitleRequest::new(session_id.clone(), "Seed DB metadata"),
                |_| Ok(()),
            )
            .await?;

        std::fs::remove_file(rollout_path)?;
        std::fs::remove_file(codex_home.join("session_index.jsonl"))?;

        agent
            .set_session_title(
                SetSessionTitleRequest::new(session_id, "DB backed title"),
                |_| Ok(()),
            )
            .await?;

        assert_eq!(
            find_thread_name_by_id(&agent.config.codex_home, &thread_id).await?,
            Some("DB backed title".to_string())
        );

        std::fs::remove_dir_all(codex_home).ok();

        Ok(())
    }

    #[tokio::test]
    async fn set_session_title_clear_notifies_with_cleared_title() -> anyhow::Result<()> {
        let codex_home =
            std::env::temp_dir().join(format!("codex-acp-clear-title-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&codex_home)?;

        let mut config = Config::load_with_cli_overrides_and_harness_overrides(
            vec![],
            ConfigOverrides::default(),
        )
        .await?;
        config.codex_home = codex_home.clone().try_into()?;
        config.model_provider_id = "test-provider".to_string();

        let agent = CodexAgent::new(config, None).await?;
        let thread_uuid = Uuid::new_v4();
        let thread_id = ThreadId::from_string(&thread_uuid.to_string())?;
        write_minimal_rollout_with_id(
            &codex_home,
            thread_uuid,
            Some("Summarize the quarterly plan"),
        )?;
        let session_id = SessionId::new(thread_id.to_string());
        let notifications = Arc::new(Mutex::new(Vec::new()));
        let captured_notifications = notifications.clone();

        agent
            .set_session_title(
                SetSessionTitleRequest::new(session_id.clone(), "   "),
                |notification| {
                    captured_notifications.lock().unwrap().push(notification);
                    Ok(())
                },
            )
            .await?;

        assert_eq!(
            codex_core::find_thread_name_by_id(&agent.config.codex_home, &thread_id).await?,
            Some("   ".to_string())
        );

        let notifications = notifications.lock().unwrap();
        assert!(
            notifications.iter().any(|notification| {
                notification.session_id == session_id
                    && matches!(
                        &notification.update,
                        SessionUpdate::SessionInfoUpdate(update)
                            if matches!(&update.title, MaybeUndefined::Null)
                    )
            }),
            "expected SessionInfoUpdate with cleared title after clearing custom title; got: {notifications:#?}"
        );
        drop(notifications);

        std::fs::remove_dir_all(codex_home).ok();

        Ok(())
    }

    #[tokio::test]
    async fn set_session_title_rejects_missing_unloaded_session() -> anyhow::Result<()> {
        let codex_home =
            std::env::temp_dir().join(format!("codex-acp-missing-title-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&codex_home)?;

        let mut config = Config::load_with_cli_overrides_and_harness_overrides(
            vec![],
            ConfigOverrides::default(),
        )
        .await?;
        config.codex_home = codex_home.clone().try_into()?;
        config.model_provider_id = "test-provider".to_string();

        let agent = CodexAgent::new(config, None).await?;
        let thread_id = ThreadId::default();
        let session_id = SessionId::new(thread_id.to_string());
        let notifications = Arc::new(Mutex::new(Vec::new()));
        let captured_notifications = notifications.clone();

        let result = agent
            .set_session_title(
                SetSessionTitleRequest::new(session_id, "Orphan title"),
                |notification| {
                    captured_notifications.lock().unwrap().push(notification);
                    Ok(())
                },
            )
            .await;

        let err = match result {
            Ok(_) => panic!("expected missing unloaded session to be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.code, acp::ErrorCode::ResourceNotFound);
        assert_eq!(
            codex_core::find_thread_name_by_id(&agent.config.codex_home, &thread_id).await?,
            None
        );
        assert!(notifications.lock().unwrap().is_empty());

        std::fs::remove_dir_all(codex_home).ok();

        Ok(())
    }

    #[tokio::test]
    async fn set_session_title_requires_authentication_before_metadata_update() -> anyhow::Result<()>
    {
        let codex_home =
            std::env::temp_dir().join(format!("codex-acp-auth-title-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&codex_home)?;

        let mut config = Config::load_with_cli_overrides_and_harness_overrides(
            vec![],
            ConfigOverrides::default(),
        )
        .await?;
        config.codex_home = codex_home.clone().try_into()?;
        config.model_provider_id = "openai".to_string();

        let agent = CodexAgent::new(config, None).await?;
        let thread_uuid = Uuid::new_v4();
        let thread_id = ThreadId::from_string(&thread_uuid.to_string())?;
        write_minimal_rollout_with_id(&codex_home, thread_uuid, None)?;
        let session_id = SessionId::new(thread_id.to_string());
        let notifications = Arc::new(Mutex::new(Vec::new()));
        let captured_notifications = notifications.clone();

        let result = agent
            .set_session_title(
                SetSessionTitleRequest::new(session_id, "Unauthenticated title"),
                |notification| {
                    captured_notifications.lock().unwrap().push(notification);
                    Ok(())
                },
            )
            .await;

        let err = match result {
            Ok(_) => panic!("expected unauthenticated title update to be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.code, acp::ErrorCode::AuthRequired);
        assert_eq!(
            find_thread_name_by_id(&agent.config.codex_home, &thread_id).await?,
            None
        );
        assert!(notifications.lock().unwrap().is_empty());

        std::fs::remove_dir_all(codex_home).ok();

        Ok(())
    }
}
