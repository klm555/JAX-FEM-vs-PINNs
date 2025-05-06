#%% L2 relative error vs. training(solving) time
import os
import json
import matplotlib.pyplot as plt

# Load evaluation results
with open(os.path.join('PINNs_evaluation.json'), 'r') as f:
    pinns_data = json.load(f)

with open('FEM_results.json', 'r') as f:
    fem_data = json.load(f)

# Slice results data
pinns_arch = pinns_data['arch']
pinns_times_train = pinns_data['times_total']
pinns_times_eval = pinns_data['times_eval']
pinns_l2_rel = pinns_data['l2_rel']

fem_ns = fem_data['mesh_nums'] # list
fem_l2_rel = fem_data['l2_rel']
fem_times_solve = fem_data['times_solve']
fem_times_eval = fem_data['times_eval']

# Plot PINNs performance
plt.figure(figsize=(7, 5))
# Annotate each point with its architecture
for idx, architecture in pinns_arch.items():
    plt.scatter(pinns_l2_rel[idx], pinns_times_train[idx], s=100, label=f"PINNs, {architecture}")

# Plot FEM performance
for idx, ns in enumerate(fem_ns):
    plt.scatter(fem_l2_rel[str(idx)], fem_times_solve[str(idx)], s=100, marker='^', label=f"FEM,    {ns}x{ns}")

plt.xlabel(r"Relative $L_2$ Error")
plt.ylabel("Training / Solving Time (seconds)")
plt.title("Relative $L_2$ Error vs. Training/Solving Time")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save figures
fig_dir = './fig'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

filename = os.path.join(fig_dir, 'solving_time-error.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()

#%% L2 relative error vs. evaluation time
import os
import json
import matplotlib.pyplot as plt

# Load evaluation results
with open(os.path.join('PINNs_evaluation.json'), 'r') as f:
    pinns_data = json.load(f)

with open('FEM_results.json', 'r') as f:
    fem_data = json.load(f)

# Slice results data
pinns_arch = pinns_data['arch']
pinns_times_train = pinns_data['times_total']
pinns_times_eval = pinns_data['times_eval']
pinns_l2_rel = pinns_data['l2_rel']

fem_ns = fem_data['mesh_nums'] # list
fem_l2_rel = fem_data['l2_rel']
fem_times_solve = fem_data['times_solve']
fem_times_eval = fem_data['times_eval']

# Plot PINNs performance
plt.figure(figsize=(7, 5))
# Annotate each point with its architecture
for idx, architecture in pinns_arch.items():
    plt.scatter(pinns_l2_rel[idx], pinns_times_eval[idx], s=100, label=f"PINNs, {architecture}")

# Plot FEM performance
for idx, ns in enumerate(fem_ns):
    plt.scatter(fem_l2_rel[str(idx)], fem_times_eval[str(idx)], s=100, marker='^', label=f"FEM,    {ns}x{ns}")

plt.xlabel(r"Relative $L_2$ Error")
plt.ylabel("Evaluation Time (seconds)")
plt.title("Relative $L_2$ Error vs. Evaluation Time")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save figures
fig_dir = './fig'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

filename = os.path.join(fig_dir, 'evaluation_time-error.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()
# %%
