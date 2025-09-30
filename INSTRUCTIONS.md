# Coveo ML Developer - NLP Challenge

> PROPRIETARY and CONFIDENTIAL, Coveo Solutions Inc.
> v2.0.0 - Last Edited 2024-10-01.

Welcome to the Coveo ML Developer take-home challenge.

## Context / Overview

Applied Scientists in your team have been working on a new proof-of-concept of an NLP model to solve a multiple choice question answering (MCQA) problem. The task that the problem solves is to fill in a missing word (`[BLANK]`) in a passage of text, given a set choice of words.

The Applied Scientists have successfully trained a promising model. Your task is to turn this into a Python project that will serve the trained model as a containerised REST API. The materials provided to you by the Applied Scientists are described in the [Artifacts Description](#artifacts-description) section, as well as instructions on how to access them.

## Expectations

What you should deliver for your teammates:

- **The Python project**. This also should include: Simple instructions on how to set up the environment, build the Docker image, serve the model, and run some test queries. A barebones codebase has been provided for you to build upon if you wish.
- **A brief operational analysis**. Operational characteristics of the service. How many resources are
required? What level of performance can we expect? This should be as an [_ANALYSIS.md_ document](ANALYSIS.md).
- **Discussion of your solution, its design, and future work**. Include a [_DESIGN.md_ document](DESIGN.md) that includes a brief explanation of your design choices, and discussion of future. This can include improvements to this project that you'd prioritise if you had more time.
  - BONUS question: How could we optimize the model for better serving performance?

If you have any questions, please do not hesitate to reach out to your recruiter (Talent Acquisition Partner) and they will get the answer to you as soon as possible.

The [Submission](#submission) section contains instructions on how to submit your solution.

## Submission

Our preference is that you submit your project as a git repository. You may do so as either:

- (Preferred:) Creating a private Github repository, and giving us the link. Ask your recruiter for the names of the Github accounts to give access to, if they have not provided them already.
- Archiving (e.g., ZIPping) your project and sending it to your recruiter.

Please talk to your recruiter if you anticipate any issues with this.

## Solution Guidance

What we're looking for:

- We expect a modern, maintainable, easy-to-understand Python project structure.
- Clean, easy-to-read code is a must.
- Please do automate any parts of the process you see as important. These are to help minimize the work needed to later move to a full CI/CD pipeline. This may, for example, include: dev environment setup, Docker building process, and necessary code quality checks and model tests.

You are not required to do anything with the training code. Do with this what you wish.

> ⚠️ NOTE: We do **not** expect you to train models. Please use the one provided.

Additional guidance:

- Treat the provided codebase as your own. Feel free to change any part of it.
- You are welcome to use diagrams and graphics in your documents.
- If you prefer to prepare your analysis (`ANALYSIS.md`) and design and future work (`DESIGN.md`) in the form of slides, you are welcome to do so. Please use no more than 5 slides.
- You may find that there are many things you'd like to change and implement in order to create a comprehensive production-worthy solution -- enough that it would take much longer than the recommended time you should spend on this. Please prioritise with what you implement. You are encouraged to note any limitations and future enhancements in `DESIGN.md`.

## Barebones Codebase

A teammate has provided you with a basic project that solves a different NLP task. You can find the project in the same directory as this instructions file. Refer to the [README.md](README.md) for the minimal documentation that comes with the codebase.

Note:

- Although this codebase is provided to you, you may wish to scrutinise it, and do not assume it is correct or ideal. Please treat it as your own. You are welcome to change any part of it when creating your solution.
- This project that your teammate's given you is solving a different problem to MCQA. Use your discretion to decide what should and shouldn't be reused for your solution to the MCQA task.

## Artifacts Description

The Python notebook(s) written by the Applied Scientists can be found in the [_notebooks_ sub-directory](notebooks).

You will also need additional files that are provided [here on Google Drive](https://drive.google.com/drive/folders/1kZRWZU_nRlBVGQlPHPH-aIMlt5C0DBZC). If you do not have access, please contact your recruiter. Please do hit _Request Access_, and then be sure to also contact your recruiter.

The Drive directory includes contains two different sets of artifacts:

1. `/data`: The datasets used to train and evaluate the model. Approx 210MB.
2. `/model`: The trained HuggingFace Pytorch model. Approx 20MB.

### The datasets (`/data`)

The folder contains 2 files: `train.csv` and `test.csv`. `train.csv` contains nine columns:

- `idx`: This is the index of the sample (an identifier for the row).
- `text`: This is a passage of text that contains `[BLANK]`. This `[BLANK]` is used to indicate a missing word.
- `choice1`, ..., `choice4`: A choice is a candidate word (a string) that could replace the `[BLANK]`.
- `label`: A string pointing to the correct choice (e.g. `choice1`, `choice2`, etc.).

`test.csv` is the same as `train.csv`, but it is missing the label.
