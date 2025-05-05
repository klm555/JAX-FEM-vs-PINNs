import json

def generate_eval_mesh(nx=10, ny=10, dt=1e-2, filename="eval_points.json"):
    """
    This function only supports:
      - 2D rectangular domain [0,1] x [0,1]
          - nx x ny         cells
          - (nx+1) x (ny+1) points
      - Time length of 1.0
    """
    # Step size in each spatial dimension
    dx = 1.0 / nx
    dy = 1.0 / ny

    # Number of time steps (incl. 0 and 1)
    nt = int(1.0 / dt) + 1
    
    # Mesh coordinates (incl. 0 and 1)
    mesh_coords = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            x = i * dx
            y = j * dx
            mesh_coords.append([x, y])
    
    # List of times (incl. 0 and 1)
    dt_coords = []
    for k in range(nt):
        t = k * dt
        dt_coords.append([t])
    
    # JSON data structure
    data = {"mesh_coord": {"0": mesh_coords},  # Store under the key "0"
            "dt_coord":   {"0": dt_coords}}   
    
    # Save
    with open(filename, "w") as f:
        json.dump(data, f, indent=2) # print JSON with better readability

if __name__ == "__main__":
    generate_eval_mesh(60, 60, 1e-2, "2D_Transient_Heat_eval_points.json")