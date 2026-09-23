import copy
import hashlib
import importlib.util
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('station', pathlib.Path(__file__).resolve().parents[1] / 'tools/spark_station.py')
station = importlib.util.module_from_spec(spec)
spec.loader.exec_module(station)


def fixture():
    model = dict(lane=1, nodes=['spark0', 'spark1'], host_mib=10, device_mib=20, pool_mib=30, ports=[[23016, 23017]], api_port=8401, resident_unit='sparkpipe-one.service', api_unit='sparkpipe-one-api.service', driver_paths=['modules/one/'], release=dict(resident_root='/release/{host}', api_root='/api', manifest_sha256={h: hashlib.sha256(b'manifest').hexdigest() for h in ['spark0', 'spark1', 'rtx5090']}))
    return dict(schema_version=1, core_commit='a'*40, api_host='rtx5090', api_user='spec', api_runtime_dir='/run/user/1000', node_limit_mib=100, weightd={h: dict(host_mib=10, device_mib=40, pool_mib=40) for h in model['nodes']}, models=dict(one=model))


class StationTests(unittest.TestCase):
    def test_budget_counts_weightd_once(self):
        s=fixture(); result=station.validate(s)
        self.assertEqual(result['spark0'], dict(reserved_mib=80, pool_mib=30))
        s['models']['two']=copy.deepcopy(s['models']['one'])
        s['models']['two'].update(lane=2, resident_unit='sparkpipe-two.service', api_unit='sparkpipe-two-api.service', api_port=8402, ports=[[23032,23033]], pool_mib=10)
        with self.assertRaisesRegex(station.StationError,'110 MiB'):station.validate(s)
        s['node_limit_mib']=110
        self.assertEqual(station.validate(s)['spark0']['pool_mib'],40)

    def test_invalid_layouts(self):
        for field,value in [('api_host','spark0'),('core_commit','main'),('node_limit_mib',79)]:
            with self.subTest(field=field):
                s=fixture();s[field]=value
                with self.assertRaises(station.StationError):station.validate(s)
        for field,value in [('lane',16),('nodes',['spark0','spark0']),('pool_mib',41),('host_mib',0),('api_port',70000),('ports',[[8,7]])]:
            with self.subTest(field=field):
                s=fixture();s['models']['one'][field]=value
                with self.assertRaises(station.StationError):station.validate(s)

    def test_port_collision(self):
        s=fixture();s['models']['one']['ports']=[[1,2],[2,3]]
        with self.assertRaisesRegex(station.StationError,'already owned'):station.validate(s)

    def test_driver_boundary_uses_real_git_diff(self):
        with tempfile.TemporaryDirectory() as d:
            def git(*args):return subprocess.check_output(['git','-C',d,*args],text=True).strip()
            git('init','-q');git('config','user.name','Test');git('config','user.email','test@example.invalid')
            p=pathlib.Path(d);(p/'modules/one').mkdir(parents=True);(p/'modules/one/driver.c').write_text('base');(p/'core.c').write_text('base')
            git('add','.');git('commit','-qm','base');s=fixture();s['core_commit']=git('rev-parse','HEAD')
            (p/'modules/one/driver.c').write_text('candidate');git('commit','-qam','driver')
            self.assertEqual(station.check_driver(d,s,'one','HEAD')['changed'],['modules/one/driver.c'])
            (p/'core.c').write_text('changed');git('commit','-qam','core')
            with self.assertRaisesRegex(station.StationError,'core.c'):station.check_driver(d,s,'one','HEAD')

    def test_manifest_drift_does_not_start_or_stop(self):
        s=fixture()
        with patch.object(station,'ssh',return_value='changed'),patch.object(station,'systemctl') as ctl,patch.object(station,'queue') as queue:
            with self.assertRaisesRegex(station.StationError,'manifest changed'):station.start(s,'one')
            ctl.assert_not_called();queue.assert_not_called()

    def exercise_start(self, failure=None):
        s=fixture();events=[]
        def ssh(host,args):
            if args[0]=='cat':return 'manifest'
            if args[0]=='python3' and 'model_residentd ready' in args[-1]:
                events.append(('ready',host))
                if failure==host:raise station.StationError('init failure '+host)
            return '{}'
        def ctl(s,m,h,*args):
            events.append((args[0],h))
            if args[0]=='show':return 'active' if ('start',h) in events else 'inactive'
        def queue(s,*args):
            events.append(('queue',args[0]))
            if args[0]=='track' and failure=='track':raise station.StationError('track failure')
            return '{}'
        with patch.object(station,'ssh',side_effect=ssh),patch.object(station,'systemctl',side_effect=ctl),patch.object(station,'queue',side_effect=queue),patch.object(station,'mesh') as mesh,patch.object(station,'stop') as stop:
            if failure:
                with self.assertRaisesRegex(station.StationError,'this family was stopped'):station.start(s,'one')
                stop.assert_called_once_with(s,'one')
                self.assertNotIn(('start','rtx5090'),events)
            else:
                station.start(s,'one');stop.assert_not_called()
                self.assertLess(events.index(('ready','spark0')),events.index(('start','rtx5090')))
                self.assertLess(events.index(('ready','spark1')),events.index(('start','rtx5090')))
            mesh.assert_called_once_with(s)

    def test_ready_barrier(self):self.exercise_start()
    def test_rank_failure_cleans_family(self):self.exercise_start('spark1')
    def test_tracking_failure_cleans_family(self):self.exercise_start('track')

    def test_stale_mesh_does_not_start(self):
        s=fixture()
        with patch.object(station,'ssh',return_value='manifest'),patch.object(station,'systemctl',return_value='inactive') as ctl,patch.object(station,'queue'),patch.object(station,'mesh',side_effect=station.StationError('stale mesh')):
            with self.assertRaisesRegex(station.StationError,'stale mesh'):station.start(s,'one')
            self.assertTrue(all(c.args[3]=='show' for c in ctl.call_args_list))

    def test_exchange_requires_stopped_models(self):
        with patch.object(station,'inspect',return_value={'units':{'spark0':{'ActiveState':'active'}}}),patch.object(station,'ssh') as ssh:
            with self.assertRaisesRegex(station.StationError,'stop all'):station.mesh(fixture(),exchange=True)
            ssh.assert_not_called()

    def test_stop_only_untracks_its_own_units(self):
        s=fixture()
        with patch.object(station,'systemctl') as ctl,patch.object(station,'tracked',return_value={'persistent:spark0:system:sparkpipe-one.service','persistent:spark0:system:sparkpipe-other.service'}),patch.object(station,'queue') as queue:
            station.stop(s,'one')
            queue.assert_called_once_with(s,'untrack','--id','persistent:spark0:system:sparkpipe-one.service')
            self.assertEqual({c.args[2] for c in ctl.call_args_list},{'spark0','spark1','rtx5090'})


if __name__=='__main__':unittest.main()
