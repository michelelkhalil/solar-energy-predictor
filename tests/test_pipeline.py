import unittest
import numpy as np
import pandas as pd
from solar.data import prepare, features, split_days, FEATURES

class PipelineTests(unittest.TestCase):
    def fixture(self):
        g=pd.DataFrame({'DATE_TIME':['15-05-2020 12:00','15-05-2020 12:00','15-05-2020 12:15'], 'PLANT_ID':[1,1,1], 'SOURCE_KEY':['a','b','a'], 'AC_POWER':[10.,20.,11.]})
        w=pd.DataFrame({'DATE_TIME':['2020-05-15 12:00:00','2020-05-15 12:15:00'], 'PLANT_ID':[1,1], 'IRRADIATION':[.8,.7], 'AMBIENT_TEMPERATURE':[30.,31.]})
        return g,w
    def test_partial_coverage_excluded(self):
        d,a=prepare(*self.fixture())
        self.assertEqual(len(d),1)
        self.assertEqual(d.AC_POWER.iloc[0],30)
        self.assertEqual(a['inverters'],2)
    def test_duplicate_rejected(self):
        g,w=self.fixture()
        with self.assertRaises(ValueError): prepare(pd.concat([g,g.iloc[:1]]),w)
    def test_nonfinite_target_excludes_timestamp(self):
        g,w=self.fixture(); g.loc[0,'AC_POWER']=np.inf
        d,_=prepare(g,w)
        self.assertEqual(len(d),0)
    def test_whole_days_and_no_temporal_overlap(self):
        d=pd.DataFrame({'DATE_TIME':pd.date_range('2020-01-01',periods=24*20,freq='h')})
        a,b,c=split_days(d)
        self.assertLess(a.DATE_TIME.max(),b.DATE_TIME.min())
        self.assertLess(b.DATE_TIME.max(),c.DATE_TIME.min())
        self.assertEqual(len(a)+len(b)+len(c),len(d))
        self.assertTrue(set(a.DATE_TIME.dt.date).isdisjoint(b.DATE_TIME.dt.date))
    def test_features_cannot_include_target_or_yield(self):
        d,_=prepare(*self.fixture()); d['DAILY_YIELD']=999
        x=features(d)
        self.assertEqual(list(x),FEATURES)
        self.assertAlmostEqual(x.hour_sin.iloc[0],0)
        self.assertAlmostEqual(x.hour_cos.iloc[0],-1)

if __name__=='__main__': unittest.main()
