"""CPU OpenUSD validation. This deliberately does not initialize Isaac Sim."""

import argparse
import hashlib
import json
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


def validate(root):
    root = Path(root).resolve()
    rows = []
    for path in sorted(root.glob("*.usda")):
        stage = Usd.Stage.Open(str(path))
        if not stage:
            raise ValueError("cannot parse " + path.name)
        prim = stage.GetDefaultPrim()
        assert prim and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert len(bodies) == 1
        colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert colliders
        for collider in colliders:
            if collider.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(collider)
                valid, reason = mesh.ValidateTopology(
                    mesh.GetFaceVertexIndicesAttr().Get(),
                    mesh.GetFaceVertexCountsAttr().Get(),
                    len(mesh.GetPointsAttr().Get()),
                )
                assert valid, reason
        reference_stage = Usd.Stage.CreateInMemory()
        reference = reference_stage.DefinePrim("/Fixture/Tool", "Xform")
        reference.GetReferences().AddReference(str(path))
        referenced_colliders = [
            p for p in Usd.PrimRange(reference) if p.HasAPI(UsdPhysics.CollisionAPI)
        ]
        assert len(referenced_colliders) == len(colliders)
        material, _ = UsdShade.MaterialBindingAPI(referenced_colliders[0]).ComputeBoundMaterial()
        assert material and str(material.GetPath()).startswith("/Fixture/Tool/Looks/")
        bounds = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
            .ComputeWorldBound(prim)
            .ComputeAlignedBox()
        )
        rows.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rigid_bodies": len(bodies),
                "colliders": len(colliders),
                "bounds_min_m": list(bounds.GetMin()),
                "bounds_max_m": list(bounds.GetMax()),
                "reference_material_binding": str(material.GetPath()),
            }
        )
    assert len(rows) == 4
    report = {
        "validation": "CPU OpenUSD parse, topology, reference composition and bounds only; native PhysX contacts/YAM NOT validated",
        "openusd_version": list(Usd.GetVersion()),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets": rows,
    }
    (root / "cpu-usd-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path(__file__).parent / "assets")
    print(json.dumps(validate(parser.parse_args().assets), indent=2))
