# Use GPT-4 API to generate code for a given prompt
import os
import openai
from .LMP import LMP


openai.organization = "org-3RBckaRMgqYfrez6l1XnkmWi"
openai.api_key = os.getenv("OPENAI_API_KEY")


# Available GPT models: gpt-4-1106-preview, gpt-4-0613, gpt-4-0314, gpt-4


lmp_planner_cfg = {
    'prompt_fname': 'planner_prompt',
    # 'model': 'gpt-4-1106-preview',  # have issues in code generation
    'model': 'gpt-4',
    # 'model': 'gpt-4o-2024-05-13',
    'max_tokens': 2048,
    'temperature': 0,
    'query_prefix': '# Query: ',
    'query_suffix': '.',
    'stop': [
        '# Query: ',
    ],
    'maintain_session': False,
    'include_context': False,
    'has_return': False,
    'return_val_name': 'ret_val',
    'load_cache': True,
    'reference_image_path': None,
}


def setup_LMP(waymo_scene: bool = False, reference_image_path: str = None):
    # Use different prompt for Waymo scenes
    if waymo_scene:
        lmp_planner_cfg['prompt_fname'] = 'planner_prompt_waymo'
    if reference_image_path:
        lmp_planner_cfg['reference_image_path'] = reference_image_path
    # Create LMP
    task_planner = LMP('planner', lmp_planner_cfg, {}, {}, debug=False, env='')
    lmps = {
        'plan_ui': task_planner,
    }
    return lmps


if __name__ == "__main__":
    # test single generate_code
    edit_instruction = "Create a traffic jam in front of me."
    lmps = setup_LMP(waymo_scene=True)
    edit_lmp = lmps['plan_ui']
    edit_lmp(edit_instruction)