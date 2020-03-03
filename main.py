from iquant3d_terminal import *
iq3t = iq3t('data','53Cr',washout=30,threshold=1E4,split=4) #data_folder, time_standard_element, washout_time, bold_width
#iq3t.run_test()
#iq3t.run_rapid()
iq3t.run(norm='23Na')

iq3t.multi_layer('13C')
iq3t.multi_layer('23Na')
iq3t.multi_layer('25Mg')
iq3t.multi_layer('39K')
iq3t.multi_layer('31P')
iq3t.multi_layer('43Ca')
iq3t.multi_layer('53Cr')
iq3t.multi_layer('55Mn')
iq3t.multi_layer('57Fe')
iq3t.multi_layer('64Zn')
iq3t.multi_layer('65Cu')
iq3t.multi_layer('77Se')
iq3t.multi_layer('95Mo')

iq3t.finish_code()
