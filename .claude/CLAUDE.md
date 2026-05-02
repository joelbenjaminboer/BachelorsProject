- Do not make any changes until you are 95% confidence in what you need to build. If you are not sure, ask for clarification.

- Bachelor's Project: "Predicting Human knee angle using a transformer-based model on IMU data"

- Architectural decisions:
  - Use a transformer-based model for time series prediction.
  - Preprocess IMU data to create input sequences and corresponding targets.
  - Train the model on the preprocessed data and evaluate its performance.
  - Using Hydra for configuration management to easily switch between different settings and parameters.

- Use small HAIKU subagents for any exploration or research and return only summarized information

### Applied learning
When something is failing or not working over and over or when a workaround is found add a one liner to the "Applied learning" section in the README.md file. Keep this under 15 words and concise. only add things that will save time in the future

    - 