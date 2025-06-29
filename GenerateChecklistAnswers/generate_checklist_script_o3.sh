
# NeurIPS 2023 accepted - o3
python generate_checklistanswers_with_separate_prompts.py --input_dir data/NeurIPS2023_latest/accepted --output_dir checklists/o3/NeurIPS2023/accepted --file_pattern "./*/structured_paper_output/paper.md" --max_workers 200 --deployment_name Checklist-GPT-o3 --debug_timer


# NeurIPS 2024 accepted - o3
python generate_checklistanswers_with_separate_prompts.py --input_dir data/NeurIPS2024_latest/accepted --output_dir checklists/o3/NeurIPS2024/accepted --file_pattern "./*/structured_paper_output/paper.md" --max_workers 400 --deployment_name Checklist-GPT-o3 --debug_timer


# ICML 2024 accepted - o3
python generate_checklistanswers_with_separate_prompts.py --input_dir data/ICML2024_latest/accepted --output_dir checklists/o3/ICML2024/accepted --file_pattern "./*/structured_paper_output/paper.md" --max_workers 400 --deployment_name Checklist-GPT-o3 --debug_timer


# ICML 2025 accepted - o3
python generate_checklistanswers_with_separate_prompts.py --input_dir data/ICML2025_latest/accepted --output_dir checklists/o3/ICML2025/accepted --file_pattern "./*/structured_paper_output/paper.md" --max_workers 400 --deployment_name Checklist-GPT-o3 --debug_timer
# ICML 2025 rejected - o3
python generate_checklistanswers_with_separate_prompts.py --input_dir data/ICML2025_latest/rejected --output_dir checklists/o3/ICML2025/rejected --file_pattern "./*/structured_paper_output/paper.md" --max_workers 400 --deployment_name Checklist-GPT-o3 --debug_timer


# ICLR 2024 accepted - o3
python generate_checklistanswers_with_separate_prompts.py --input_dir data/ICLR2024_latest/accepted --output_dir checklists/o3/ICLR2024/accepted --file_pattern "./*/structured_paper_output/paper.md" --max_workers 400 --deployment_name Checklist-GPT-o3 --debug_timer
# ICLR 2024 rejected - o3
python generate_checklistanswers_with_separate_prompts.py --input_dir data/ICLR2024_latest/rejected --output_dir checklists/03/ICLR2024/rejected --file_pattern "./*/structured_paper_output/paper.md" --max_workers 400 --deployment_name Checklist-GPT-o3 --debug_timer


# ICLR 2025 accepted - o3
python generate_checklistanswers_with_separate_prompts.py --input_dir data/ICLR2025_latest/accepted --output_dir checklists/o3/ICLR2025/accepted --file_pattern "./*/structured_paper_output/paper.md" --max_workers 400 --deployment_name Checklist-GPT-o3 --debug_timer
# ICLR 2025 rejected - o3
python generate_checklistanswers_with_separate_prompts.py --input_dir data/ICLR2025_latest/rejected --output_dir checklists/03/ICLR2025/rejected --file_pattern "./*/structured_paper_output/paper.md" --max_workers 400 --deployment_name Checklist-GPT-o3 --debug_timer



