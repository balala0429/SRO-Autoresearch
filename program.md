# Autonomous Research - Symbolic Regression Optimization (SRO)

This is an experiment to have the AI Coding Agent do its own autonomous research on a Symbolic Regression Optimization (SRO) solver.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `train.py` (formerly solve_residual.py) — The main solver loop, OMP fitting, and retrieval logic.
   - `Predictor/ResidualPredictor.py` — The neural network architecture that maps residuals to vector representations.
4. **Verify data/weights exist**: Ensure `library/library/symbolic_index.bin` and `weight2/predictor_final.pth` exist. 
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good and kick off experimentation.

## Experimentation

**What you CAN do:**
- Modify `train.py` — specifically the `ResidualSolver` logic, retrieval constraints, component splicing, and OMP fitting.
- Modify `Predictor/ResidualPredictor.py` — architecture improvements for the neural network.

**The goal is simple: get the lowest Final MSE on the target dataset (Nguyen-5).** 

### Primary Research Directives (The TODOs)
The user has left specific areas for you to explore and optimize. Focus your experiments on these:
1. **Improve Retrieval Logic**: Modify the FAISS search and Beam Search logic to ensure component quality. Try injecting initially extracted subtree components into the candidate pool to maintain high-quality priors.
2. **Enhance ResidualPredictor**: Modify the neural network structure in `ResidualPredictor.py` (or redefine it) to better map the residual vectors `y_input` to the latent space `v_pred`. Higher capacity or better architectural priors might improve the FAISS retrieval accuracy.
3. **Advanced Splicing Operators**: Currently, components are combined via linear least squares (OMP). Try introducing non-linear splicing operators. For example, pass the extracted components through neural network activation functions (e.g., Sigmoid, ReLU, Tanh) before or during the global refitting process to capture extreme non-linearities like `sin(x^2)`.

**Simplicity criterion**: All else being equal, simpler is better. If a complex modification only yields a tiny MSE improvement (e.g., 0.0001), discard it. If an improvement simplifies the tree structure or reduces the number of required patches, definitely keep it.

**The first run**: Your very first run should always be to establish the baseline. Run the script as is without modifications.

## Output format

The training script evaluates the final formula and prints the mean squared error at the very end. Look for this exact string in the logs:

##最终 MSE: [VALUE]

*(e.g., `最终 MSE: 0.123456`)*

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 4 columns:

##commit   mse      status   description


1. git commit hash (short, 7 chars)
2. mse achieved (e.g. 0.054321) — use 999.0 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

##Example:

commit   mse      status   description
a1b2c3d  0.245000 keep     baseline
b2c3d4e  0.150000 keep     add ReLU activation before OMP linear combination
c3d4e5f  0.251000 discard  increase Predictor hidden dimension to 512
d4e5f6g  999.000  crash    modify FAISS search k to 1000 (OOM)


## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` or `ResidualPredictor.py` with an experimental idea based on the Primary Research Directives.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (or `uv run train.py > run.log 2>&1` depending on environment). Do NOT use tee.
5. Read out the results: `grep "最终 MSE:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the `results.tsv` (NOTE: do not commit `results.tsv`).
8. If MSE improved (lower), you "advance" the branch, keeping the git commit.
9. If MSE is equal or worse, you git reset back to where you started (`git reset --hard HEAD~1`).

**Timeout**: SRO fitting should be fast. If a single run exceeds 3 minutes, kill it and treat it as a failure (discard and revert).

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try radically different activation functions, rethink the beam search scoring penalty, or alter the `max_iters` dynamically. The loop runs until the human interrupts you, period.