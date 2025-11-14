# Mem-PAL

The data and code of the paper "Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction" (AAAI 2026 Oral).

The Appendix of this paper is present in [here](Appendix_of_Mem-PAL.pdf).

## PAL-Set Synthesis

### Dataset

The PAL-Set can be found in the `./data_synthesis_v2/data/input.json` file. The dataset consists of a total of 100 users' dicts, with the keys being user ids. Each user dict contains two parts: `history` and `query`, which represent the history set and query set, respectively. Each set contains multiple samples, where:

- `sample_id`: The id number of each sample.
- `dialogue_timestamp`: The timestamp when the current sample dialogue occurred.
- `logs`: A list containing all logs under the current sample, including the log timestamps `timestamp` and log contents `content`. The list is ordered chronologically by the log timestamps, and all log timestamps occur earlier than the timestamp of the current dialogue.
- `dialogue`: The multi-turn dialogue corresponding to the current sample. Each turn of the dialogue includes one user utterance and one assistant reply. Each user or assistant utterance contains the dialogue action `action` and the utterance content `content`.
- `topics`: The dialogue framework corresponding to each dialogue, which may include 1~3 topics. Each topic in the query set can serve as an evaluation sample for single-turn QA tasks (Requirement Restatement or Solution Proposal). The content of each topic includes:
    - `user_query`: The initial vague user query of the topic.
    - `implicit_needs`: A list containing 2 entries that are not explicitly mentioned in the user query but are intended to be inferred by the assistant.
    - `requirement`: A detailed requirement description combined the user query and implicit needs in a sentence.
    - `solution`: Contains two solutions that best align with the user's preferences (`pos`) and two solutions that least align with the user's preferences (`neg`).
    - `candidate_solutions`: All 8 candidate solutions, each labeled with the user's feedback type.

In addition, each user's background, trait, and situations are also provided in the `./data_synthesis_v2/data` directory.

The current version of the PAL-Set is in Chinese. We plan to create and release an additional English-translated version soon.

### Data Synthesis Pipeline

The code for the data synthesis process can be found in the ./data_synthesis_v2/code directory, where:

- 1\) Background: `background.py`
- 2\) Global Persona: `global_persona.py`
- 3\) Situation & 4\) Experience: `situation.py`
- 5\) Dialogue Framework: `dialogue_framework.py`
- 6\) Logs & Dialogues: `log.py` & `dialogue.py`

Examples and templates of prompts for each step can be found in the `./data_synthesis_v2/prompt_template` directory.


## H<sup>2</sup>Memory Framework & Evaluation

The codes for constructing the H<sup>2</sup>Memory and performing each task using that memory on our PAL-Bench can be found in `./perassist_framework/code/memory_rag_v2/`, while the corresponding prompts can be found in `./perassist_framework/prompt_template/memory_rag_v2/`.

### H<sup>2</sup>Memory Framework

- 1\) Log Graph: `log_analysis.py`
- 2\) Background: `background_summary.py`
- 3\) Topic Outline: `dialogue_analysis.py`
- 4\) Principle: `pinciple_extraction.py`


### Evaluation

#### Requirement Restatement

- code: `requirement_prediction_mamory_rag_v2.py`
- GPT-4 Evaluation Prompt: `./perassist_framework/code/evaluation/requirement_prediction_evaluation_prompt`

#### Solution Proposal

Solution Generation:
- code: `solution_qa_mamory_rag_v2.py`

Solution Selection:
- code: `solution_selection_mamory_rag_v2.py`

#### Multi-turn Dialogue Interaction

- code: `dialogue_interaction_memory_rag_v2.py`
- The prompt of User-LLM and Evaluation-LLM of this task can be found in `./perassist_framework/code/dialogue_evaluation`

