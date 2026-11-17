"""Static necessary condition for the frozen stop, with rigid contacts and fixed frame."""
import json
import math
from pathlib import Path


def evaluate(geometry, opening_degrees=90.):
    # At a centred stop, the two jaws rotate through equal opposite angles.
    half=math.radians(opening_degrees-3)/2
    jaw_mass=.060
    com_x,com_z=.012,.025
    piston_mass=.030
    gravity=9.81
    jaw_rate=geometry["native_joint_parameters"]["jaw_total_angle_rad_per_m"]/2
    derivative=2*jaw_mass*gravity*jaw_rate*(com_x*math.cos(half)-com_z*math.sin(half))+piston_mass*gravity
    thread=geometry["thread_model"]
    ks=thread["axial_stiffness_N_per_m"]["KS"]
    kn=thread["axial_stiffness_N_per_m"]["KN"]
    lead=thread["p_m_per_rad"]
    permitted_z=1e-6
    permitted_turn=math.radians(.01)
    # For rigid nut seating and zero steady speed, subtract unloaded and loaded
    # vertical force equilibrium. Give both angle excursions the helpful sign.
    max_load=(ks+kn)*permitted_z+(ks+kn+kn)*lead*permitted_turn
    minimum_open_port=1.
    minimum_load=minimum_open_port-derivative
    return dict(scope="ideal rigid seat, fixed frame, settled static comparison; separate from GPU traces",
                target_open_deg=opening_degrees,jaw_gravity_plus_piston_open_effort_n=-derivative,
                minimum_settled_stop_load_n=minimum_load,max_force_change_in_pose_window_n=max_load,
                minimum_axial_change_um=1e6*(minimum_load-(ks+2*kn)*lead*permitted_turn)/(ks+kn),
                nominal_5n_settled_stop_load_n=5.-derivative,
                compatible_with_static_window=minimum_load<=max_load)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("geometry",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    a.out.write_text(json.dumps(evaluate(json.loads(a.geometry.read_text())),indent=2)+"\n")
