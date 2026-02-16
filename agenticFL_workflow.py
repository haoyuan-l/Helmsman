from planning_graph import create_planning_workflow
from coding_graph import create_coding_workflow
from langchain_core.messages import HumanMessage, ToolMessage
import os
import getpass
from typing import Dict, Any, List, Optional
import asyncio
from langgraph.types import Command
import sys

# ============================================================================
# CONFIGURATION & MODEL DEFINITIONS
# ============================================================================

AVAILABLE_MODELS = {
    "planning": {
        "1": {"name": "gemini-2.0-flash-exp", "display": "Gemini 2.0 Flash Experimental (Fast, Cost-Effective)", "provider": "google"},
        "2": {"name": "gemini-2.5-flash", "display": "Gemini 2.5 Flash (Balanced Performance)", "provider": "google"},
        "3": {"name": "gemini-2.5-pro", "display": "Gemini 2.5 Pro (Most Capable)", "provider": "google"},
    },
    "coding": {
        "1": {"name": "gpt-4.1", "display": "GPT-4.1 (Reliable)", "provider": "openai"},
        "2": {"name": "gpt-5.1", "display": "GPT-5.1 (Advanced)", "provider": "openai"},
        "3": {"name": "claude-sonnet-4-5-20250929", "display": "Claude Sonnet 4.5 (Code Expert)", "provider": "anthropic"},
        "4": {"name": "gemini-2.5-pro", "display": "Gemini 2.5 Pro (Versatile)", "provider": "google"},
    }
}

API_KEY_MAP = {
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "tavily": "TAVILY_API_KEY",
}

# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================

def print_banner():
    """Display welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║        🤖 FEDERATED LEARNING AGENTIC SYSTEM 🤖                   ║
║                                                                   ║
║     Automated Research Planning & Code Generation System         ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_step(step_num: int, title: str):
    """Print a step indicator."""
    print(f"\n{'─' * 70}")
    print(f"📍 STEP {step_num}: {title}")
    print('─' * 70)

def get_valid_input(prompt: str, valid_options: List[str], allow_empty: bool = False) -> str:
    """Get validated user input."""
    while True:
        user_input = input(prompt).strip()
        
        if allow_empty and user_input == "":
            return user_input
            
        if user_input in valid_options:
            return user_input
        
        print(f"❌ Invalid input. Please choose from: {', '.join(valid_options)}")

def confirm_action(message: str) -> bool:
    """Ask for user confirmation."""
    response = get_valid_input(f"\n{message} (yes/no): ", ["yes", "y", "no", "n"])
    return response in ["yes", "y"]

# ============================================================================
# MODEL SELECTION
# ============================================================================

def select_model(workflow_type: str) -> Dict[str, Any]:
    """Interactive model selection for a specific workflow."""
    print_step(1 if workflow_type == "planning" else 2, 
               f"Select {workflow_type.upper()} Model")
    
    models = AVAILABLE_MODELS[workflow_type]
    
    print(f"\n🎯 Available models for {workflow_type}:\n")
    for key, model_info in models.items():
        print(f"  [{key}] {model_info['display']}")
    
    choice = get_valid_input("\n👉 Enter your choice: ", list(models.keys()))
    selected = models[choice]
    
    print(f"\n✅ Selected: {selected['display']}")
    return selected

def display_model_summary(planning_model: Dict, coding_model: Dict):
    """Display a summary of selected models."""
    print_section_header("MODEL CONFIGURATION SUMMARY")
    print(f"\n  📋 Planning Agent:  {planning_model['display']}")
    print(f"                      └─ Model: {planning_model['name']}")
    print(f"\n  💻 Coding Agent:    {coding_model['display']}")
    print(f"                      └─ Model: {coding_model['name']}")

# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def get_required_api_keys(planning_model: Dict, coding_model: Dict) -> List[str]:
    """Determine which API keys are needed based on selected models."""
    required_providers = set()
    
    # Add providers for selected models
    required_providers.add(planning_model['provider'])
    required_providers.add(coding_model['provider'])
    
    # Always need these for tools/utilities
    always_required = ["voyage", "cohere", "tavily"]
    
    required_keys = []
    for provider in required_providers:
        if provider in API_KEY_MAP:
            required_keys.append(API_KEY_MAP[provider])
    
    for provider in always_required:
        key = API_KEY_MAP[provider]
        if key not in required_keys:
            required_keys.append(key)
    
    return required_keys

def check_existing_api_keys(required_keys: List[str]) -> Dict[str, bool]:
    """Check which API keys are already set."""
    status = {}
    for key in required_keys:
        status[key] = bool(os.environ.get(key))
    return status

def setup_api_keys(planning_model: Dict, coding_model: Dict):
    """Smart API key setup - only ask for what's needed."""
    print_step(3, "API Key Configuration")
    
    required_keys = get_required_api_keys(planning_model, coding_model)
    existing_status = check_existing_api_keys(required_keys)
    
    # Display status
    print("\n🔑 API Key Status:\n")
    all_set = True
    for key in required_keys:
        status_icon = "✅" if existing_status[key] else "❌"
        status_text = "Set" if existing_status[key] else "Missing"
        print(f"  {status_icon} {key}: {status_text}")
        if not existing_status[key]:
            all_set = False
    
    if all_set:
        print("\n✅ All required API keys are already configured!")
        return True
    
    print("\n⚠️  Some API keys are missing.")
    
    if not confirm_action("Would you like to enter them now?"):
        print("\n❌ Cannot proceed without required API keys.")
        return False
    
    # Prompt for missing keys
    print("\n📝 Please enter the missing API keys:\n")
    for key in required_keys:
        if not existing_status[key]:
            while True:
                value = getpass.getpass(f"  🔐 {key}: ").strip()
                if value:
                    os.environ[key] = value
                    print(f"  ✅ {key} saved")
                    break
                else:
                    if confirm_action(f"  ⚠️  Skip {key}? (May cause errors)"):
                        break
    
    print("\n✅ API key configuration complete!")
    return True

# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

def run_planning_workflow(graph, thread_id="planning_session_1"):
    """Execute the planning workflow with detailed state tracking."""
    
    thread_config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "user_query": None,
        "messages": None,
        "user_feedback": None,
        "current_plan": None,
        "plan_approved": False,
        "plan_status": None
    }

    print_section_header("🚀 PLANNING WORKFLOW STARTED")
    
    # Initial invocation
    state = graph.invoke(initial_state, config=thread_config)
    
    iteration = 0
    while True:
        snapshot = graph.get_state(thread_config)
        
        if not snapshot.next:
            print("\n" + "=" * 70)
            print("✅ Planning Workflow Complete!")
            print("=" * 70)
            
            # Display final results
            if state.get('current_plan'):
                print(f"\n📋 FINAL APPROVED RESEARCH PLAN")
                print("─" * 70)
                print(state['current_plan'])
                print("─" * 70)
            
            if state.get('retrieved_content'):
                print(f"\n📚 Retrieved {len(state['retrieved_content'])} research documents")
                
            break
        
        # Show current state info
        next_node = snapshot.next[0]
        current_values = snapshot.values
        iteration += 1
        
        print(f"\n🔄 Iteration {iteration} - Current Node: {next_node}")
        
        # Display relevant state information
        if current_values.get('messages'):
            print(f"   📨 Messages in state: {len(current_values['messages'])}")
        if current_values.get('planning_history'):
            print(f"   📚 Planning iterations: {len(current_values['planning_history'])}")
        if current_values.get('plan_status'):
            print(f"   📊 Plan status: {current_values['plan_status']}")
        
        # Handle different interrupt types
        if next_node == "human_query":
            print("\n" + "=" * 70)
            print("🎯 INITIAL QUERY REQUIRED")
            print("=" * 70)
            print("\nPlease describe your federated learning research question.")
            print("Example: 'Create a federated learning system for image classification")
            print("         using the MNIST dataset with privacy-preserving techniques'\n")
            
            user_input = input("📝 Your research question:\n> ").strip()
            
            if not user_input:
                print("❌ Empty input. Please provide a research question.")
                continue
            
            state = graph.invoke(
                Command(resume=user_input),
                config=thread_config
            )
            
        elif next_node == "human_decision":
            print("\n" + "=" * 70)
            print("📋 PLAN REVIEW & APPROVAL")
            print("=" * 70)
            print("\n📄 Current Research Plan:")
            print("─" * 70)
            plan_text = current_values.get('current_plan', 'No plan available')
            print(plan_text)
            print("─" * 70)
            
            decision = get_valid_input("\n👉 Approve this plan? (yes/no): ", ["yes", "no"])
            
            state = graph.invoke(
                Command(resume=decision),
                config=thread_config
            )
            
        elif next_node == "human_feedback":
            print("\n" + "=" * 70)
            print("🔄 PLAN REFINEMENT REQUIRED")
            print("=" * 70)
            
            if current_values.get('agent_reflection'):
                print("\n💭 Agent Analysis:")
                print("─" * 70)
                print(current_values['agent_reflection'])
                print("─" * 70)
            
            print("\nPlease provide additional requirements or clarifications to improve the plan.")
            print("Examples:")
            print("  - 'Add more focus on privacy-preserving techniques'")
            print("  - 'Include differential privacy mechanisms'")
            print("  - 'Use a different dataset like CIFAR-10'\n")
            
            feedback = input("📝 Your feedback:\n> ").strip()
            
            if not feedback:
                print("⚠️  No feedback provided. Using empty feedback.")
            
            state = graph.invoke(
                Command(resume={"user_feedback": feedback}),
                config=thread_config
            )
            
        else:
            print(f"⚠️ Unexpected state: {next_node}")
            break
    
    return state

def run_coding_workflow(graph, research_plan, thread_id="coding_session_1"):
    """Execute the coding workflow based on the research plan."""
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100
    }    
    
    initial_state = {
        "research_plan": research_plan,
        "is_optimized": False,
    }
    
    print_section_header("🚀 CODING WORKFLOW STARTED")
    print("\n⏳ This process may take 10-20 minutes as the system:")
    print("   • Generates FL module code")
    print("   • Writes comprehensive tests")
    print("   • Debugs and fixes issues")
    print("   • Runs complete FL simulations")
    print("\n☕ Grab a coffee and relax...\n")
    
    simulation_attempts = 0
    max_simulation_attempts = 10
    
    try:
        # Stream through the graph execution
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                print(f"\n{'─' * 70}")
                print(f"📍 Node: {node_name}")
                print('─' * 70)
                
                # Display relevant information based on node type
                if node_name == "supervisor":
                    print("✅ Supervisor analyzed research plan and created module tasks")
                    if "implementation_overview" in node_output:
                        print("📋 Implementation overview generated")
                
                elif "coder" in node_name:
                    module_name = node_name.replace("_coder", "").replace("_", " ").title()
                    print(f"💻 Implementing {module_name}...")
                    
                elif "test" in node_name and node_name != "orchestrator_test":
                    module_name = node_name.replace("_test", "").replace("_", " ").title()
                    pass_status = node_output.get(f"{node_name.replace('_test', '')}_pass_status", False)
                    if pass_status:
                        print(f"✅ {module_name} - All tests passed!")
                    else:
                        print(f"🔧 {module_name} - Debugging in progress...")
                
                elif node_name == "orchestrator_node":
                    print("🎯 Creating orchestration script (run.py)...")
                
                elif node_name == "orchestrator_test":
                    if node_output.get("run_pass_status", False):
                        print("✅ Orchestrator test passed! Ready for simulation...")
                    else:
                        print("🔧 Orchestrator debugging in progress...")
                
                elif node_name == "evaluator":
                    simulation_attempts += 1
                    print(f"🚀 FL Simulation Running (Attempt {simulation_attempts}/{max_simulation_attempts})")
                    print("📊 Executing 3 rounds of federated learning...")
                    print("⏳ Please wait...")
                    
                elif node_name == "simulation_debugger":
                    print(f"🔧 Debugging runtime errors (Attempt {simulation_attempts}/{max_simulation_attempts})")
                    print("📝 Analyzing errors and applying fixes...")
                    
                    if simulation_attempts >= max_simulation_attempts:
                        print(f"\n⚠️  Maximum simulation attempts ({max_simulation_attempts}) reached.")
                        print("Manual intervention may be required.")
                        break
        
        # Get the final state
        final_state = graph.get_state(config).values
        
        print("\n" + "=" * 70)
        print("✅ CODING WORKFLOW COMPLETED")
        print("=" * 70)
        
        # Display summary of generated files
        print("\n📁 Generated Codebase (fl_codebase/):\n")
        files_generated = []
        
        file_map = {
            "codebase_task": ("task.py", "Model, training, and data loading"),
            "codebase_client": ("client_app.py", "FL Client implementation"),
            "codebase_server": ("server_app.py", "FL Server configuration"),
            "codebase_strategy": ("strategy.py", "Custom FL strategy"),
            "codebase_run": ("run.py", "Orchestration script")
        }
        
        for state_key, (filename, description) in file_map.items():
            if final_state.get(state_key):
                print(f"   ✅ {filename:20s} - {description}")
        
        # Check final simulation status
        print("\n" + "=" * 70)
        if final_state.get("run_pass_status", False):
            print("🎉 SUCCESS: Federated Learning System Ready!")
            print("=" * 70)
            print("\n📊 The FL simulation completed all 3 rounds successfully!")
            print("\n🚀 Next Steps:")
            print("   1. Review the code: cd fl_codebase/")
            print("   2. Run simulation:  python run.py")
            print("   3. Customize:       Edit files as needed")
        else:
            print("⚠️  WARNING: Simulation Issues Detected")
            print("=" * 70)
            print("\nThe FL system was generated but encountered runtime issues.")
            print("Please review error messages and consider manual debugging.")
            
        # Display simulation statistics
        if final_state.get("run_test_feedback"):
            print("\n📊 Simulation Feedback:")
            print("─" * 70)
            feedback = final_state["run_test_feedback"]
            print(feedback[:500] + "..." if len(feedback) > 500 else feedback)
            
            if simulation_attempts >= max_simulation_attempts:
                print(f"\n⚠️  Note: {simulation_attempts} debugging attempts were made")
        
        return final_state
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user.")
        print("Progress has been saved. You can resume if the system supports it.")
        return None
        
    except Exception as e:
        print(f"\n❌ Error in coding workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow with interactive setup."""
    try:
        # Welcome banner
        print_banner()
        print("\nWelcome! This system will help you:")
        print("  1️⃣  Plan your federated learning research")
        print("  2️⃣  Generate complete FL code implementation")
        print("  3️⃣  Test and validate the system")
        
        # Step 1: Select planning model
        planning_model = select_model("planning")
        
        # Step 2: Select coding model
        coding_model = select_model("coding")
        
        # Display summary
        display_model_summary(planning_model, coding_model)
        
        # Step 3: Setup API keys
        if not setup_api_keys(planning_model, coding_model):
            print("\n❌ Setup cancelled. Exiting.")
            return
        
        # Final confirmation
        print_section_header("READY TO START")
        print("\n✨ Configuration complete! The system will now:")
        print("   • Create a detailed research plan (with your guidance)")
        print("   • Generate FL code for all modules")
        print("   • Test and debug the implementation")
        print("   • Run a complete FL simulation")
        
        if not confirm_action("\n🚀 Start the workflow?"):
            print("\n👋 Workflow cancelled. Goodbye!")
            return
        
        # Create workflows
        print("\n⏳ Initializing workflows...")
        
        plan_workflow = create_planning_workflow(
            plan_model_name=planning_model['name'],
        )
        
        code_workflow = create_coding_workflow(
            code_model_name=coding_model['name'],
        )
        
        print("✅ Workflows initialized successfully!")
        
        # Run planning workflow
        final_planning_state = run_planning_workflow(plan_workflow)
        research_plan = final_planning_state.get("current_plan", "")
        
        if not research_plan:
            print("\n❌ No research plan generated. Cannot proceed to coding.")
            return
        
        # Confirm before coding
        print_section_header("TRANSITION TO CODING")
        print("\n✅ Research plan is ready!")
        
        if not confirm_action("Proceed to code generation?"):
            print("\n💾 Research plan saved. You can use it later for code generation.")
            print("👋 Exiting. Goodbye!")
            return
        
        # Run coding workflow
        final_coding_state = run_coding_workflow(code_workflow, research_plan)
        
        if final_coding_state:
            print_section_header("🎉 ALL WORKFLOWS COMPLETED!")
            print("\n✅ Your federated learning system is ready!")
            print("\n📁 Check the fl_codebase/ directory for all generated files.")
            print("📖 Review the code and run: python fl_codebase/run.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user. Exiting gracefully...")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()