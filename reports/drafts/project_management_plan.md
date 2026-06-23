# Project Management Plan

## Scope

This project is scoped as a seven-week capstone prototype. The final deliverables are a written report, GitHub codebase, and recorded presentation. The prototype will focus on a complete but manageable machine learning workflow: data preparation, EDA, customer journey modeling, baseline comparison, model optimization, evaluation, and a Streamlit demo app.

## Weekly Plan

| Week | Course Milestone | Project Work |
| --- | --- | --- |
| 1 | Topic, dataset, introduction draft | Finalize project framing, dataset choice, research question, and introduction draft. |
| 2 | Project management plan | Define scope, tasks, risks, evaluation criteria, and repository structure. |
| 3 | Data cleaning and EDA | Run `scripts/01_data_cleaning_eda.py`, generate figures, and draft data summary. |
| 4 | Background and method selection | Review customer journey modeling, sequence models, uplift/activation literature, and agentic marketing systems. |
| 5 | Begin model training and methods draft | Train LSTM and baselines with `scripts/02_model_training_evaluation.py`. Document methods. |
| 6 | Complete model work and start results | Run optimization with `scripts/03_model_optimization.py`, generate evaluation artifacts, and analyze pipeline demo cases. |
| 7 | Final report, presentation, GitHub repo | Polish README, report, presentation visuals, app demo, and repository organization. |

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Public dataset download or parsing issues | Keep synthetic retail simulator as fallback and document public-data intent. |
| Model labels are weak supervision labels | Explain this limitation clearly and propose campaign-response labels as future work. |
| Mistral local model may be slow | Keep deterministic explanation fallback and make Ollama optional. |
| Project scope grows too large | Keep focus on audience-to-campaign fit, not rebuilding SegmentAI or a full production marketing platform. |

## Success Criteria

- Working Streamlit app.
- Deep learning model trained from scratch.
- EDA figures and dataset summary.
- Baseline model comparison.
- Hyperparameter optimization evidence.
- Representative input/output examples.
- Clear report and presentation visuals.
