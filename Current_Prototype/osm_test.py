import os
import openstudio

from bca import _paths_config

repo_path = _paths_config.repo_root
osm_name = r'Current_Prototype/BEM/OpenStudioModels/BEM_5z_2A_Base_Testbed.osm'


osm_path = openstudio.path(os.path.join(repo_path, osm_name))

t = True
if t:
    translator = openstudio.osversion.VersionTranslator()
    osm = translator.loadModel(osm_path).get()
else:
    osm = openstudio.model.Model.load(osm_path).get()


ts = osm.getTimestep()
print(ts)
ts.setNumberOfTimestepsPerHour(4)
print(ts)

run_period = osm.getRunPeriod()
print(run_period)
run_period.setEndMonth(1)
print(run_period)

fwd_translator = openstudio.energyplus.ForwardTranslator()
out_idf = fwd_translator.translateModel(osm)
out_idf.save(openstudio.path('test_idf.idf'), True)


in_osm = openstudio.energyplus.ReverseTranslator('test_idf.idf')