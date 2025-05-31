import math, random as rn, time, copy
import noise
from datetime import datetime
import os

import numpy as np
import open3d as o3d
from scipy import interpolate
from scipy.interpolate import BSpline

import visualization

# ───────────────────────────────────────────────────────────────────────────────
# Parameters
# ───────────────────────────────────────────────────────────────────────────────

GROUND_SIZE       = 100.0     # side length in (roughly) meters
GROUND_RES        = 0.05      # grid spacing for mesh triangles (starts to slow down at 0.025) in (roughly) meters
GROUND_N_POINTS   = 300000    # how many pts to sample for the preview cloud

# Perlin‑noise params
GROUND_NOISE_FREQ      = 0.5 #0.1 "width of bumps"
GROUND_NOISE_OCTAVES   = 15
GROUND_NOISE_AMPLITUDE = 0.05   # ± m about z=0

# Track generation
MIN_POINTS, MAX_POINTS = 19, 20
DIFFICULTY             = 0.3
SPLINE_POINTS          = 5000
TRACK_WIDTH            = 5.0
SAFETY_MARGIN          = 5.0

# Objects
CONE_RADIUS = 0.3
CONE_HEIGHT = 0.4

# Environment bulding
CONE_SPACING = 5.0     # how frequently along the track cones are placed

# Lidar
LIDAR_HEIGHT     = 1
V_ANGLES         = np.linspace(-10, 10, 50)
H_ANGLES         = np.linspace(30, 150, 300)
N_SCAN_POS       = 50
TOTAL_ENVIRONMENTS_TO_GENERATE = 20

# Lidar scan point cloud visualization
LSPCV_POINT_SIZE = 4
FRAME_DELAY = 0.1


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v

# ───────────────────────────────────────────────────────────────────────────────
# Track‑generation helpers (Circular Control Point Algorithm with Aberration)
# ───────────────────────────────────────────────────────────────────────────────

def _circular_control_points():
    """Generate control points placed equidistantly on a circle with a controlled
    radial aberration applied to each. The points are scaled to comfortably fit
    inside the allowable ground area (accounting for margins and safety
    clearances). The ordering of the points is preserved, guaranteeing a simple
    non‑self‑intersecting polygon.
    """
    # 1. Choose the control‑point count.
    n = rn.randrange(MIN_POINTS, MAX_POINTS + 1)

    # 2. Evenly distribute angles around the circle.
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)

    # 3. Generate a bounded radial perturbation for each point. Using DIFFICULTY
    #    as the maximum percentage deviation from the unit‑circle radius keeps
    #    the aberration gentle enough to avoid self‑intersections while still
    #    producing visually interesting waviness.
    aberr = (np.random.rand(n) * 2.0 - 1.0) * DIFFICULTY  # range [‑D, +D]
    radii  = 1.0 + aberr                                # range [1‑D, 1+D]

    # 4. Convert polar coordinates to Cartesian.
    unit_pts = np.stack((np.cos(angles) * radii, np.sin(angles) * radii), axis=1)

    # 5. Scale the unit‑circle polygon so that its farthest extent sits well
    #    inside the ground. "half" is the maximum allowable half‑width (ground
    #    half‑size minus margin and a bit of headroom).
    half = GROUND_SIZE / 2.0 - SAFETY_MARGIN - TRACK_WIDTH
    unit_max = np.max(np.linalg.norm(unit_pts, axis=1))
    scale = half / unit_max
    pts = unit_pts * scale

    return pts


def _smooth(pts):
    """Build and return a periodic BSpline object for these control points."""
    # Close the loop explicitly
    x = np.r_[pts[:,0], pts[0,0]]
    y = np.r_[pts[:,1], pts[0,1]]
    # Get knots, coeffs and degree
    tck, _ = interpolate.splprep([x, y], s=0, per=True)
    knots, coeffs, degree = tck
    # coeffs is [c_x, c_y], stack into shape (n_coeffs, 2)
    coeffs = np.vstack(coeffs).T
    # Build a periodic BSpline
    spline = BSpline(knots, coeffs, degree, extrapolate='periodic')
    return spline


def _fit_to_bounds(cl):
    x_max = np.max(cl[:,0]) + TRACK_WIDTH/2
    x_min = np.min(cl[:,0]) - TRACK_WIDTH/2
    y_max = np.max(cl[:,1]) + TRACK_WIDTH/2
    y_min = np.min(cl[:,1]) - TRACK_WIDTH/2
    half = GROUND_SIZE/2 - SAFETY_MARGIN
    max_extent = max(abs(x_max), abs(x_min), abs(y_max), abs(y_min))
    if max_extent > half:
        scale = half / max_extent
        cl = cl * scale
    return cl


def generate_centerline(bspline=None):
    """
    1. Create control points with radial aberration.
    2. Smooth them with a periodic B‑spline to obtain a dense centre‑line.
    3. Boundary‑fit (typically a no‑op thanks to the initial scale).
    """
    print("Starting centerline generation (circular control‑point algorithm)...")

    if bspline is None:
        control_pts = _circular_control_points()
        spline      = _smooth(control_pts)
    else:
        spline      = bspline
    print("Built periodic BSpline object.")

    # Sample the spline densely for boundary fitting
    param       = np.linspace(0, 1, SPLINE_POINTS)
    smooth_cl   = spline(param)
    fitted_cl   = _fit_to_bounds(smooth_cl)

    print("Centerline generation complete.\n")

    return fitted_cl

# ───────────────────────────────────────────────────────────────────────────────
# Ground‑mesh generator
# ───────────────────────────────────────────────────────────────────────────────

def make_ground(res: float = GROUND_RES) -> o3d.geometry.TriangleMesh:
    print('1')
    xs = np.arange(-GROUND_SIZE/2,  GROUND_SIZE/2 + res, res)
    ys = np.arange(-GROUND_SIZE/2,  GROUND_SIZE/2 + res, res)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pnoise_vec = np.vectorize(
        lambda x, y: noise.pnoise2(
            x * GROUND_NOISE_FREQ,
            y * GROUND_NOISE_FREQ,
            octaves=GROUND_NOISE_OCTAVES,
        ),
        otypes=[float],
    )
    zz = pnoise_vec(xx, yy) * GROUND_NOISE_AMPLITUDE
    verts = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    ny, nx = zz.shape
    faces = []

    # simple text‐progress bar
    bar_len = 30
    for r in range(ny - 1):
        # update bar
        filled = int(bar_len * (r+1) / (ny - 1))
        bar = '=' * filled + ' ' * (bar_len - filled)
        print(f'\rbuilding ground mesh: [{bar}] {r+1}/{ny-1}', end='', flush=True)

        base = r * nx
        for c in range(nx - 1):
            idx = base + c
            faces.append([idx,     idx + 1,         idx + nx])
            faces.append([idx + 1, idx + nx + 1,    idx + nx])
    print()  # newline when done

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    mesh.paint_uniform_color([0.35, 0.35, 0.35])
    mesh.compute_vertex_normals()
    print('2')
    return mesh

# ───────────────────────────────────────────────────────────────────────────────
# Cone helper
# ───────────────────────────────────────────────────────────────────────────────

def cone_mesh():
    m = o3d.geometry.TriangleMesh.create_cone(radius=CONE_RADIUS, height=CONE_HEIGHT, resolution=20)
    m.paint_uniform_color([1,0.55,0])
    m.compute_vertex_normals()
    return m

# ───────────────────────────────────────────────────────────────────────────────
# Environment creation
# ───────────────────────────────────────────────────────────────────────────────

def build_environment():
    ground = make_ground()
    center = generate_centerline()
    print("Centerline generated")

    base = cone_mesh()
    cones = []
    arc_len = np.sum(np.linalg.norm(center[1:] - center[:-1], axis=1))
    spacing = CONE_SPACING
    steps = max(1, int(arc_len // spacing))
    idxs = np.linspace(0, len(center) - 1, steps, dtype=int)
    for i in idxs:
        prev = center[i-1]
        nxt  = center[(i+1) % len(center)]
        norm = _unit(np.array([-(nxt - prev)[1], (nxt - prev)[0]]))
        for off in (+1, -1):
            pos = center[i] + off * norm * (TRACK_WIDTH / 2)
            c = copy.deepcopy(base)
            c.translate([pos[0], pos[1], 0])
            cones.append(c)

    env_pcd = ground.sample_points_uniformly(GROUND_N_POINTS)
    for c in cones:
        env_pcd += c.sample_points_uniformly(300)

    print('3')

    return [ground, *cones], env_pcd, center

# ───────────────────────────────────────────────────────────────────────────────
# Raycasting helpers
# ───────────────────────────────────────────────────────────────────────────────

def build_scene(meshes):
    scene = o3d.t.geometry.RaycastingScene()
    total   = len(meshes)
    bar_len = 40
    for i, m in enumerate(meshes, start=1):
        filled = int(bar_len * i / total)
        bar    = '=' * filled + ' ' * (bar_len - filled)
        print(f'\rbuilding raycasting scene: [{bar}] {i}/{total}', end='', flush=True)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    print()  # newline when done
    return scene


def rays(v_ang, h_ang):
    v = np.deg2rad(v_ang)[:, None]
    h = np.deg2rad(h_ang)[None, :]
    x = np.cos(v) * np.cos(h)
    y = np.cos(v) * np.sin(h)
    z = np.sin(v) * np.ones_like(x)
    d = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)

RAW_DIRS = rays(V_ANGLES, H_ANGLES).astype(np.float32)

# ───────────────────────────────────────────────────────────────────────────────
# Lidar scan helper
# ───────────────────────────────────────────────────────────────────────────────

def scan(scene, pos, tangent, h=LIDAR_HEIGHT):
    yaw = math.atan2(tangent[1], tangent[0])
    yaw_offset = yaw - (math.pi/2)
    c, s = math.cos(yaw_offset), math.sin(yaw_offset)
    R = np.array([[ c, -s, 0],
                  [ s,  c, 0],
                  [ 0,  0, 1]], dtype=np.float32)

    dirs = (RAW_DIRS @ R.T)
    origin = np.array([pos[0], pos[1], h], dtype=np.float32)
    origins = np.repeat(origin[None,:], dirs.shape[0], axis=0)
    ray_bundle = np.hstack((origins, dirs))

    hits = scene.cast_rays(o3d.core.Tensor(ray_bundle))
    d = hits['t_hit'].numpy()
    mask = np.isfinite(d)
    if not np.any(mask):
        return o3d.geometry.PointCloud(), np.empty(0, dtype=np.uint8)

    pts_world = origins[mask] + dirs[mask] * d[mask][:,None]
    geom_ids = hits['geometry_ids'].numpy()[mask]
    labels   = (geom_ids != 0).astype(np.uint8)  # 0=ground, 1=cone
    rel_world = pts_world - np.array([pos[0], pos[1], 0], dtype=np.float32)[None,:]
    rel = rel_world @ R

    col = np.zeros((rel.shape[0], 3), dtype=np.float32)
    col[:, 2] = 1.0  # blue

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rel))
    pc.colors = o3d.utility.Vector3dVector(col)
    return pc, labels

# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def main():
    all_generated_scans = []
    all_generated_labels = []
    
    # Timestamp for the entire batch of environments
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder_name = f"lidar_scans_batch_{batch_timestamp}"
    os.makedirs(base_folder_name, exist_ok=True)

    for env_num in range(TOTAL_ENVIRONMENTS_TO_GENERATE):
        print(f"\n--- Generating Environment {env_num + 1} / {TOTAL_ENVIRONMENTS_TO_GENERATE} ---")
        # build_environment() will generate a new random track each time
        meshes, pcd, center = build_environment() #
        
        # create red spline overlay (optional, for visualization if you were to use it per environment)
        # pts3d = np.hstack((center, np.zeros((center.shape[0], 1), dtype=np.float64)))
        # lines = [[i, (i+1) % len(center)] for i in range(len(center))]
        # line_set = o3d.geometry.LineSet(
        #     points=o3d.utility.Vector3dVector(pts3d),
        #     lines=o3d.utility.Vector2iVector(lines)
        # )
        # line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])
        # print('4')

        scene = build_scene(meshes) #

        # N_SCAN_POS now defines scans per environment/track
        idxs = np.linspace(0, len(center)-1, N_SCAN_POS, dtype=int) #
        
        current_env_scans = []
        current_env_labels = []
        
        total_scans_in_env = len(idxs)
        bar_len = 40
        print(f"Generating {total_scans_in_env} scans for environment {env_num + 1}...")
        for j, i in enumerate(idxs, start=1): #
            prev = center[i-1]
            nxt  = center[(i+1) % len(center)]
            tangent = _unit(nxt - prev)

            pc, lab = scan(scene, np.append(center[i], 0.0), tangent) #
            # Store points relative to the scan, as done by your scan() function
            all_generated_scans.append(np.asarray(pc.points).astype(np.float32))
            all_generated_labels.append(lab.astype(np.int32))

            filled = int(bar_len * j / total_scans_in_env)
            bar = '=' * filled + ' ' * (bar_len - filled)
            print(f'\rgenerating scans for env {env_num + 1}: [{bar}] {j}/{total_scans_in_env}', end='', flush=True)
        print() # Newline after progress bar for current environment

    print(f"\n--- All environments processed. Total scans generated: {len(all_generated_scans)} ---")

    # Save all collected scans
    # You might want to adjust the folder structure or naming convention
    # For instance, save each environment's scans in a subfolder,
    # or ensure unique filenames if all are in one folder.
    # This example saves them all sequentially in the batch folder.
    print(f"Saving {len(all_generated_scans)} lidar scans to folder {base_folder_name}...")
    for idx, (pts, lab) in enumerate(zip(all_generated_scans, all_generated_labels)):
        np.savez_compressed(
            os.path.join(base_folder_name, f"scan_{idx:04d}.npz"), # Increased padding for more scans
            points=pts,
            labels=lab
        )
    print(f"All lidar scan data saved to folder {base_folder_name}")

if __name__ == '__main__':
    main()
