import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory


class FailureReceiptTests(unittest.TestCase):
    def test_material_readbacks_use_float_tolerance_not_exact_equality(self):
        path=Path(__file__).with_name('run_full_screen_yam_preflight.py')
        tree=ast.parse(path.read_text())
        material_checks=[]
        for node in ast.walk(tree):
            if not isinstance(node,ast.Dict): continue
            fields={key.value:value for key,value in zip(node.keys,node.values)
                    if isinstance(key,ast.Constant) and isinstance(key.value,str)}
            name=fields.get('name')
            if (isinstance(name,ast.Constant) and isinstance(name.value,str)
                    and 'friction_readback' in name.value):
                material_checks.append(fields)
        self.assertEqual(len(material_checks),4)
        for fields in material_checks:
            self.assertEqual(fields['operator'].value,'<=')
            self.assertAlmostEqual(fields['threshold'].value,1e-6)
            self.assertIn('absolute_error',fields['name'].value)

    def test_primary_survives_each_double_failure_finalizer(self):
        for failing_phase in ('video_writer.close','trace_path.write_text','env.close','simulation_app.close'):
            with self.subTest(failing_phase=failing_phase):
                ns=self.load_finalizer_helpers()
                try:
                    raise RuntimeError('original motion failure')
                except RuntimeError:
                    primary=ns['sys'].exc_info()
                secondary=[]
                def fail(): raise OSError('injected '+failing_phase)
                operations=[(name,fail if name==failing_phase else (lambda:None)) for name in
                    ('video_writer.close','trace_path.write_text','env.close','simulation_app.close')]
                saved=ns['run_finalizers_preserving_primary'](primary,operations,secondary)
                self.assertIs(saved[1],primary[1])
                self.assertEqual(secondary,[{'phase':failing_phase,'type':'OSError',
                                             'message':'injected '+failing_phase}])
                with self.assertRaisesRegex(RuntimeError,'original motion failure') as raised:
                    ns['raise_primary'](saved)
                self.assertIs(raised.exception,primary[1])

    def test_first_cleanup_failure_survives_later_cleanup_failure(self):
        ns=self.load_finalizer_helpers(); secondary=[]
        def writer_fail(): raise ValueError('writer primary')
        def env_fail(): raise OSError('env secondary')
        saved=ns['run_finalizers_preserving_primary'](None,
            [('video_writer.close',writer_fail),('env.close',env_fail)],secondary)
        self.assertIsInstance(saved[1],ValueError)
        self.assertEqual(secondary[0]['phase'],'env.close')
        with self.assertRaisesRegex(ValueError,'writer primary'):
            ns['raise_primary'](saved)

    def test_failed_postprocessing_preserves_video_trace_and_measured_fields(self):
        with TemporaryDirectory() as folder:
            workspace=Path(folder)/'workspace'; (workspace/'task').mkdir(parents=True)
            measurements=workspace/'measurements'; measurements.mkdir()
            report=measurements/'report.json'; video=measurements/'video.mp4'
            shot=measurements/'shot.png'; trace=measurements/'report-trace.json'
            video.write_bytes(b'fixture-video-bytes'); shot.write_bytes(b'fixture-image-bytes')
            trace.write_bytes(b'[{"control_sample":19,"native_z":0.84}]')
            report.write_text(json.dumps({'native_mass_kg':.4,'checks':[]}))
            before={p:p.read_bytes() for p in (video,shot,trace)}
            def bad_main(): raise RuntimeError('injected postprocessing failure')
            ns=self.load(workspace,report,video,shot,bad_main)
            with self.assertRaisesRegex(RuntimeError,'injected postprocessing'):
                ns['run_with_failure_receipt']()
            saved=json.loads(report.read_text())
            self.assertFalse(saved['component_preflight_passed'])
            self.assertEqual(saved['native_mass_kg'],.4)
            self.assertEqual(saved['execution_exception']['type'],'RuntimeError')
            self.assertFalse(saved['checks'][-1]['passed'])
            for p,data in before.items(): self.assertEqual(p.read_bytes(),data)

    def test_success_does_not_rewrite_report(self):
        with TemporaryDirectory() as folder:
            workspace=Path(folder); report=workspace/'report.json'; report.write_text('original bytes')
            ns=self.load(workspace,report,workspace/'v.mp4',workspace/'s.png',lambda:42)
            self.assertEqual(ns['run_with_failure_receipt'](),42)
            self.assertEqual(report.read_text(),'original bytes')

    @staticmethod
    def load(workspace,report,video,shot,main):
        path=Path(__file__).with_name('run_full_screen_yam_preflight.py')
        node=next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='run_with_failure_receipt')
        ns={'json':json,'Path':Path,'__file__':str(workspace/'task/runner.py'),
            'args':SimpleNamespace(report=report,video=video,screenshot=shot),'main':main,
            'simulation_app':SimpleNamespace(close=lambda:None)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        return ns

    @staticmethod
    def load_finalizer_helpers():
        import sys
        path=Path(__file__).with_name('run_full_screen_yam_preflight.py')
        wanted={'run_finalizers_preserving_primary','raise_primary'}
        nodes=[n for n in ast.parse(path.read_text()).body
               if isinstance(n,ast.FunctionDef) and n.name in wanted]
        ns={'sys':sys}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        return ns


if __name__=='__main__':unittest.main()
