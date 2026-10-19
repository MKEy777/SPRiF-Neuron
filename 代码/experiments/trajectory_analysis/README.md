# Slow-State Trajectory Analysis

## Objective
Visualize the core design principle of SPRiF: that spikes do not reset the slow state.

## Dataset
- PS-MNIST (784-step sequence, most intuitive visualization)

## Procedure
1. Load trained SPRiF and LIF models (PS-MNIST)
2. For one test sample, record per timestep:
   - SPRiF: slow state x_t (3D trajectory) + fast state u_t membrane potential + spike times
   - LIF: membrane potential trajectory + spike times
3. Align to spike events (t = 0 at spike time)
4. Generate comparison figure:
   - **Top row**: SPRiF slow state — **continuous, no jump** at spike times
   - **Bottom row**: LIF membrane potential — **reset to zero** at spike times

## Expected Results
- SPRiF slow state remains continuous across spike events
- LIF membrane potential resets after each spike

## Output
- `trajectory_comparison.png` — SPRiF vs LIF pre/post spike comparison
