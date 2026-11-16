"""CPU source-geometry regression check; no simulator import or physics claim."""
import ast
import hashlib
import json
import math
from pathlib import Path

from shapely.geometry import box
from shapely.geometry.polygon import orient

path = Path(__file__).with_name('full_screen_scene.py')
source = path.read_text()
tree = ast.parse(source)
functions = {'complete_holes', 'disk', 'floor_pieces', 'polygon_integrals',
             'mass_properties', 'geometry_audit'}
nodes = [n for n in tree.body if
         (isinstance(n, ast.Import) and all(a.name == 'math' for a in n.names)) or
         (isinstance(n, ast.ImportFrom) and n.module and n.module.startswith('shapely')) or
         isinstance(n, (ast.Assign, ast.Assert)) or
         (isinstance(n, ast.FunctionDef) and n.name in functions)]
ns = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
for sign in (-1, 1):
    values = ns['polygon_integrals'](orient(box(-1, -2, 1, 2), sign=sign))
    assert all(math.isclose(a, b, abs_tol=1e-10)
               for a,b in zip(values, (8, 0, 0, 8/3, 32/3, 0)))
    ring = box(-2,-2,2,2).difference(box(-1,-1,1,1))
    assert math.isclose(ns['polygon_integrals'](orient(ring, sign))[0], 12)
pieces, solid = ns['floor_pieces']()
audit = ns['geometry_audit']()
assert audit['hole_count'] == 261
assert audit['solid_connected_components'] == 1
assert audit['symmetric_difference_area_m2'] < 1e-12
assert all(not p.interiors and p.symmetric_difference(p.convex_hull).area < 1e-12
           for p in pieces)
assert math.isclose(sum(p.area for p in pieces), solid.area, abs_tol=1e-12)
props = ns['mass_properties']()
assert math.isclose(props['mass'], .4)
assert props['volume'] > 0
assert all(math.isfinite(v) and v > 0 for v in props['inertia'])
assert props['inertia'][0] + props['inertia'][1] >= props['inertia'][2]
assert -ns['FLOOR_T'] < props['com'][2] < ns['DEPTH']
assert all(shape.geom_type == 'Polygon' for shape,_,_ in props['layers'])
assert props['layers'][0][0].covers(ns['disk'](ns['OUTER_R']).exterior)
print(json.dumps({'scope':'CPU source geometry only; native qualification false',
                  'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                  'mass_kg':props['mass'], 'inertia_kg_m2':props['inertia'],
                  'com_m':props['com'], 'effective_density_kg_m3':props['effective_density_kg_m3'],
                  'geometry_audit':audit}))
