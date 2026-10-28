//! Per-iteration tool-spec assembly for the turn engine.

use crate::tools::{ActivatedToolSet, Tool, ToolSpec};
use anyhow::Result;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use zeroclaw_providers::ModelProvider;

/// Tool specs assembled for one loop iteration.
pub(crate) struct IterationToolSpecs {
    // Complete admitted schemas serve parsing and argument recovery, even
    // when a built-in schema is hidden from the provider request.
    pub(crate) tool_specs: Vec<ToolSpec>,
    pub(crate) exposed_tool_specs: Option<Vec<ToolSpec>>,
    pub(crate) known_tool_names: HashSet<String>,
    pub(crate) use_native_tools: bool,
}

impl IterationToolSpecs {
    /// Count before known-tool filtering or execution preparation can hide a batch.
    pub(crate) fn exceeds_single_tool_round(
        &self,
        response: &zeroclaw_providers::ChatResponse,
        strict_tool_parsing: bool,
    ) -> bool {
        exceeds_single_tool_round(
            response.text_or_empty(),
            response.tool_calls.len(),
            !self.tool_specs.is_empty(),
            strict_tool_parsing,
        )
    }

    pub(crate) fn request_tool_specs(&self) -> &[ToolSpec] {
        self.exposed_tool_specs
            .as_deref()
            .unwrap_or(&self.tool_specs)
    }

    pub(crate) fn refresh_native_tool_mode(
        &mut self,
        model_provider: &dyn ModelProvider,
        model: &str,
    ) {
        self.use_native_tools = model_provider
            .capabilities_for_model(model)
            .native_tool_calling
            && !self.request_tool_specs().is_empty();
    }
}

pub(super) fn exceeds_single_tool_round(
    text: &str,
    native_calls: usize,
    has_tools: bool,
    strict_tool_parsing: bool,
) -> bool {
    if native_calls > 0 {
        return native_calls > 1;
    }
    if !has_tools || strict_tool_parsing {
        return false;
    }
    let text = zeroclaw_tool_call_parser::strip_think_tags(text);
    let text = zeroclaw_tool_call_parser::strip_trailing_terminal_markers(&text);
    !zeroclaw_tool_call_parser::looks_like_tool_protocol_example(&text)
        && zeroclaw_tool_call_parser::parse_tool_calls(&text).1.len() > 1
}

/// Resolve turn filtering from the current MCP owners, not name syntax or the
/// original sealed registry. The activation map owns only MCP executables;
/// deferred built-in schemas live in a separate metadata map.
pub(super) async fn excluded_mcp_tools_for_iteration(
    tools_registry: &crate::tools::scoped::ScopedToolRegistry,
    activated_tools: Option<&Arc<Mutex<ActivatedToolSet>>>,
    groups: &[zeroclaw_config::schema::ToolFilterGroup],
    user_message: &str,
) -> Vec<String> {
    if groups.is_empty() {
        return Vec::new();
    }
    let mut names = tools_registry.current_mcp_tool_names().await;
    // These reserved host names are the generic MCP capability wrappers,
    // minted only by scoped MCP assembly (never skill-owned names).
    names.extend(
        tools_registry
            .iter()
            .filter(|tool| matches!(tool.name(), "mcp_resources" | "mcp_prompts"))
            .map(|tool| tool.name().to_owned()),
    );
    if let Some(activated) = activated_tools {
        let activated = activated.lock().unwrap_or_else(|error| error.into_inner());
        names.extend(activated.tool_names().into_iter().map(str::to_owned));
    }
    let message = user_message.to_ascii_lowercase();
    let mut excluded: Vec<_> = names
        .into_iter()
        .filter(|name| !crate::agent::loop_::mcp_tool_included_for_turn(name, groups, &message))
        .collect();
    excluded.sort_unstable();
    excluded
}

pub(crate) fn build_iteration_tool_specs(
    model_provider: &dyn ModelProvider,
    model: &str,
    tools_registry: &[Box<dyn Tool>],
    excluded_tools: &[String],
    activated_tools: Option<&Arc<Mutex<ActivatedToolSet>>>,
) -> Result<IterationToolSpecs> {
    // Rebuild tool_specs each iteration so newly activated deferred tools appear.
    let mut tool_specs: Vec<crate::tools::ToolSpec> = tools_registry
        .iter()
        .filter(|tool| !excluded_tools.iter().any(|ex| ex == tool.name()))
        .map(|tool| tool.spec())
        .collect();
    let mut hidden_builtin_names = HashSet::new();
    if let Some(at) = activated_tools {
        let activated_tools = match at.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                ::zeroclaw_log::record!(
                    WARN,
                    ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                        .with_category(::zeroclaw_log::EventCategory::Tool)
                        .with_outcome(::zeroclaw_log::EventOutcome::Unknown),
                    "activated-tool lock poisoned while assembling iteration tool specs; recovering guard for read"
                );
                poisoned.into_inner()
            }
        };
        // Turn exclusions can remove search without mutating the scoped registry.
        if tool_specs.iter().any(|spec| spec.name == "tool_search") {
            hidden_builtin_names = activated_tools.hidden_builtin_names();
        }
        for spec in activated_tools.tool_specs() {
            if !excluded_tools.iter().any(|ex| ex == &spec.name) {
                // Eager MCP definition updates overlay a sealed base schema.
                // Keep a single provider definition, with the refreshed schema.
                if let Some(base) = tool_specs.iter_mut().find(|base| base.name == spec.name) {
                    *base = spec;
                } else {
                    tool_specs.push(spec);
                }
            }
        }
    }
    // >>> herdr-prompt-cache: deterministic tool ordering
    // Tool specs are assembled from unordered maps (the MCP registry keeps
    // tools in a HashMap), so their order varies between processes. The tools
    // block is the first section of every provider request, so any reshuffle
    // changes the token prefix and invalidates prompt caching for the whole
    // request (tools + system + messages) - cache reads become full cache
    // writes. Sorting by name makes the prefix byte-stable across processes.
    tool_specs.sort_by(|a, b| a.name.cmp(&b.name));
    let known_tool_names: HashSet<String> = tool_specs
        .iter()
        .map(|tool| tool.name.to_ascii_lowercase())
        .collect();
    let exposed_tool_specs = (!hidden_builtin_names.is_empty()).then(|| {
        tool_specs
            .iter()
            .filter(|spec| !hidden_builtin_names.contains(&spec.name))
            .cloned()
            .collect::<Vec<_>>()
    });
    let use_native_tools = model_provider
        .capabilities_for_model(model)
        .native_tool_calling
        && !exposed_tool_specs
            .as_deref()
            .unwrap_or(&tool_specs)
            .is_empty();

    Ok(IterationToolSpecs {
        tool_specs,
        exposed_tool_specs,
        known_tool_names,
        use_native_tools,
    })
}

#[cfg(test)]
mod tests {
    use super::build_iteration_tool_specs;
    use crate::tools::ActivatedToolSet;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use zeroclaw_api::attribution::Role;
    use zeroclaw_api::model_provider::{ModelProvider, ProviderCapabilities};
    use zeroclaw_api::tool::Tool;

    struct NativeToolsProvider;

    #[async_trait]
    impl ModelProvider for NativeToolsProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                native_tool_calling: true,
                ..ProviderCapabilities::default()
            }
        }

        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            unreachable!("test provider should not execute chat")
        }
    }

    impl zeroclaw_api::attribution::Attributable for NativeToolsProvider {
        fn role(&self) -> Role {
            Role::System
        }

        fn alias(&self) -> &str {
            "test-native-tools-provider"
        }
    }

    struct PromptToolsProvider;

    #[async_trait]
    impl ModelProvider for PromptToolsProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                native_tool_calling: false,
                ..ProviderCapabilities::default()
            }
        }

        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: Option<f64>,
        ) -> anyhow::Result<String> {
            unreachable!("test provider should not execute chat")
        }
    }

    impl zeroclaw_api::attribution::Attributable for PromptToolsProvider {
        fn role(&self) -> Role {
            Role::System
        }

        fn alias(&self) -> &str {
            "test-prompt-tools-provider"
        }
    }

    struct CountingTool {
        name: String,
        invocations: Arc<AtomicUsize>,
    }

    impl CountingTool {
        fn new(name: &str, invocations: Arc<AtomicUsize>) -> Self {
            Self {
                name: name.to_string(),
                invocations,
            }
        }
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Counts executions for poisoned-lock tests"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                }
            })
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            Ok(crate::tools::ToolResult {
                success: true,
                output: "counted".into(),
                error: None,
            })
        }
    }

    impl zeroclaw_api::attribution::Attributable for CountingTool {
        fn role(&self) -> Role {
            Role::Tool(zeroclaw_api::attribution::ToolKind::Plugin)
        }

        fn alias(&self) -> &str {
            self.name()
        }
    }

    #[test]
    fn mid_turn_mcp_filter_message_ignores_tool_carriers_memory_and_trim_breadcrumbs() {
        use crate::agent::turn::{MemoryPreamble, current_filter_user_message};
        use zeroclaw_providers::ChatMessage;

        let injected = MemoryPreamble {
            preamble: "recalled PCB context\n".into(),
            index: 3,
        };
        let history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("PCB breadcrumb"),
            ChatMessage::user("old PCB request"),
            ChatMessage::user("recalled PCB context\ncurrent request"),
            ChatMessage::assistant("PCB narration"),
            ChatMessage::user("[Tool results]\nPCB tool output"),
        ];
        assert_eq!(
            current_filter_user_message(&history, true, Some(&injected)),
            "current request"
        );
        assert_eq!(current_filter_user_message(&history[..2], true, None), "");
        assert_eq!(
            current_filter_user_message(&history[..2], false, None),
            "PCB breadcrumb"
        );
        // The streamed owner may admit steering before the first model round;
        // filtering follows that input rather than the original ingress text.
        let mut steered = history;
        steered.push(ChatMessage::user("inspect PCB"));
        assert_eq!(
            current_filter_user_message(&steered, true, Some(&injected)),
            "inspect PCB"
        );
    }

    #[test]
    fn deferred_builtin_request_projection_preserves_callable_names_and_exclusions() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let calendar = Box::new(CountingTool::new("calendar", Arc::clone(&invocations)));
        let blocked = Box::new(CountingTool::new("blocked", Arc::clone(&invocations)));
        let search = Box::new(CountingTool::new("tool_search", invocations));
        let mut activated = ActivatedToolSet::new();
        activated.set_deferred_builtin_specs(vec![calendar.spec(), blocked.spec()]);
        let activated = Arc::new(Mutex::new(activated));
        let registry: Vec<Box<dyn Tool>> = vec![calendar, blocked, search];
        let specs = build_iteration_tool_specs(
            &NativeToolsProvider,
            "test-model",
            &registry,
            &["blocked".into()],
            Some(&activated),
        )
        .unwrap();

        assert_eq!(
            specs
                .request_tool_specs()
                .iter()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            vec!["tool_search"]
        );
        assert_eq!(
            specs
                .tool_specs
                .iter()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            vec!["calendar", "tool_search"]
        );
        assert!(specs.known_tool_names.contains("calendar"));
        assert!(!specs.known_tool_names.contains("blocked"));

        let eager = build_iteration_tool_specs(
            &NativeToolsProvider,
            "test-model",
            &registry,
            &["blocked".into(), "tool_search".into()],
            Some(&activated),
        )
        .unwrap();
        assert_eq!(
            eager
                .request_tool_specs()
                .iter()
                .map(|spec| spec.name.as_str())
                .collect::<Vec<_>>(),
            vec!["calendar"]
        );
        assert!(eager.known_tool_names.contains("calendar"));
        assert!(!eager.known_tool_names.contains("blocked"));
        assert!(!eager.known_tool_names.contains("tool_search"));
        assert!(
            activated
                .lock()
                .unwrap()
                .hidden_builtin_names()
                .contains("calendar")
        );
    }

    #[test]
    fn build_iteration_tool_specs_recovers_poisoned_activated_tool_lock() {
        let activated = Arc::new(Mutex::new(ActivatedToolSet::new()));
        let invocations = Arc::new(AtomicUsize::new(0));
        let activated_tool: Arc<dyn Tool> = Arc::new(CountingTool::new(
            "docker-mcp__extract_text",
            Arc::clone(&invocations),
        ));
        activated
            .lock()
            .unwrap()
            .activate("docker-mcp__extract_text".into(), activated_tool);
        let poisoned = Arc::clone(&activated);
        let _ = std::thread::spawn(move || {
            let _guard = poisoned.lock().expect("test mutex should lock");
            panic!("poison activated-tools lock");
        })
        .join();

        let specs = build_iteration_tool_specs(
            &NativeToolsProvider,
            "test-model",
            &[],
            &[],
            Some(&activated),
        )
        .expect("poisoned activated-tools lock should recover for read");
        assert!(
            specs
                .tool_specs
                .iter()
                .any(|spec| spec.name == "docker-mcp__extract_text"),
            "recovered poisoned lock should still expose activated tool specs"
        );
    }

    #[test]
    fn iteration_tool_specs_recomputes_native_mode_for_active_provider() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let tool = Box::new(CountingTool::new("read_file", invocations));
        let mut specs =
            build_iteration_tool_specs(&NativeToolsProvider, "test-model", &[tool], &[], None)
                .expect("native provider with tools should build specs");
        assert!(specs.use_native_tools);

        specs.refresh_native_tool_mode(&PromptToolsProvider, "test-model");

        assert!(
            !specs.use_native_tools,
            "active provider must decide whether this turn uses native tool transport"
        );
    }
}
