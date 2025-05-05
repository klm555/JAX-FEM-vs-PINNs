import os
import json
from dolfin import *
import numpy as np

# Load evaluation points
with open('2D_Transient_Heat_eval_points.json', 'r') as f:
    data = json.load(f)

# Evaluation coordinates & time
mesh_coords = data["mesh_coord"]["0"]
dt_coords = data["dt_coord"]["0"] # [[0.0], [0.1], ..., [1.0]]
times = [item[0] for item in dt_coords]  # unpack to [0.0, 0.1, ...]

# Number of time intervals
nt_steps = len(times) - 1


# Mesh
ns = 200
mesh = UnitSquareMesh(ns, ns) # unit square [0,1] x [0,1]

# B.Cs
def boundary(x, on_boundary): # All boundary points
    return on_boundary

# Initial Condition
# u_0 = exp(-50*((x-0.5)^2 + (y-0.5)^2))
class InitialCondition(UserExpression): # "UserExpression" allows to define a function w/ python, unlike "Expression".
    def eval(self, values, x): # x : 2D point
        values[0] = np.exp(-50.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    def value_shape(self):
        return ()

# Function space
V = FunctionSpace(mesh, "P", 1)  # Initialize function space
# Piecewise linear finite element space

# Dirichlet BC
bc = DirichletBC(V, Constant(0.0), boundary) # u = "0" at x = "boundary"

# Initial condition
u_n = Function(V) # u_n : solution at time step n
u_n.interpolate(InitialCondition()) # assign u_0(x,y) to u_n at t=0
#    "interpolate" 
# 1. calls "eval" from InitialCondition
# 2. evaluates at all points in the mesh (mesh is assigned to "V")
# 3. assigns the result to u_n
# (while "project" requires assembly & linear solver, )

# Initialize time & time step
t = Constant(0.0) # it can be updated, even if it is a Constant
dt = Constant(0.0)

# Source term (will be updated in time loop)
# f(x,y,t) = 10*sin(pi*x)*sin(pi*y)*cos(2*pi*t)
f = Expression("10.0*sin(pi*x[0])*sin(pi*x[1])*cos(2.0*pi*t)", degree = 2, t=t) # assign t to "t" in the expression

# Trial and test functions
u = Function(V) # shouldn't it be TrialFunction(V)?
v = TestFunction(V) # it is just a symbolic variable to build the linear system

# Weak form
# R = (u - u_n)*v + dt*(grad(u), grad(v)) = dt*f_expr*v
R = (u-u_n)*v*dx + dt*dot(grad(u), grad(v))*dx - dt*f*v*dx

# Jacobian for the nonlinear solver (though this PDE is linear in u)
jac = derivative(R, u) # dF/du(jacobian) w/ symbolic derivative

# Save t=0 solution (pvd)
save_dir = './vtu'
os.makedirs(save_dir, exist_ok=True)
File(f"{save_dir}/solution_000.pvd") << u_n

# Save t=0 solution (json)
sol_list = []
sol0 = []
for (x, y) in mesh_coords:
    u_0 = u_n(Point(x, y))
    sol0.append(u_0)
sol_list.append(sol0)

# Time-stepping loop (Backward Euler)
for n in range(nt_steps):
    t_prev = times[n] # t_(n-1)
    t_curr = times[n+1] # t_n

    # Update dt
    dt.assign(t_curr - t_prev)

    # Update t (source term is also automatically updated w/ this line)
    t.assign(t_curr)

    # Solve
    solve(R == 0, u, bc, J=jac)

    # Save solution (pvd)
    if (n + 1) % 1 == 0: # "% 10" is for every 10th step
        print(f"Time step {n + 1}, t = {t_curr:.4f}")
        File(f"{save_dir}/solution_{n + 1:03d}.pvd") << u
    
    # Save solution (json)
    sol = []
    for (x, y) in mesh_coords:
        sol.append(u(Point(x, y)))
    sol_list.append(sol)

    # Assign new solution(u) into previous solution(u_n) for the next step
    u_n.assign(u)

sol_json = "2D_Transient_Heat_eval_solutions.json"
with open(sol_json, 'w') as f:
    json.dump(sol_list, f)

print(f"Solution created: {sol_json}")