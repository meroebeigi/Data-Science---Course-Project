import unittest
import json
import numpy as np
import pandas as pd
from src.experiment import ROOT, FEATURES, check_dynamics, regression_metrics


class ExperimentValidation(unittest.TestCase):
    def test_planner_matches_gym(self):
        check_dynamics()

    def test_episode_groups_are_disjoint(self):
        data=pd.read_csv(ROOT/'results/demonstrations.csv')
        self.assertEqual(data.groupby('episode').split.nunique().max(),1)
        self.assertEqual(data.groupby('split').episode.nunique().to_dict(), {'train':22,'validation':7,'test':7})
        self.assertEqual(len(data),7200)

    def test_recorded_metrics(self):
        predictions=pd.read_csv(ROOT/'results/test_predictions.csv')
        metrics=pd.read_csv(ROOT/'results/regression_metrics.csv')
        for name, frame in predictions.groupby('model'):
            expected=metrics[(metrics.model==name)&(metrics.split=='test')].iloc[0]
            for key,value in regression_metrics(frame.actual,frame.predicted).items():
                self.assertAlmostEqual(value,expected[key],places=10)

    def test_matched_controller_initial_conditions(self):
        data=pd.read_csv(ROOT/'results/controller_episodes.csv')
        for _,frame in data.groupby(['scenario','episode']):
            self.assertEqual(frame.controller.nunique(),6)
            self.assertEqual(frame.initial_angle_deg.nunique(),1)
            self.assertEqual(frame.initial_velocity.nunique(),1)

    def test_exported_neural_policy(self):
        with np.load(ROOT/'results/policy_weights.npz',allow_pickle=False) as archive:
            weights={key:archive[key] for key in archive.files}
        data=pd.read_csv(ROOT/'results/demonstrations.csv')
        frame=data[data.split=='test']
        x=(frame[FEATURES].to_numpy()-weights['mean'])/weights['scale']
        for i in [0,1]:x=np.tanh(x@weights[f'weight_{i}']+weights[f'bias_{i}'])
        prediction=np.clip((x@weights['weight_2']+weights['bias_2']).ravel(),-2,2)
        expected=pd.read_csv(ROOT/'results/test_predictions.csv')
        # Training/predict uses float32; exported forward pass uses float64.
        # Allow 0.00005 torque units (0.00125% of the actuator range).
        np.testing.assert_allclose(prediction,expected.loc[expected.model=='Neural network','predicted'],atol=5e-5)


if __name__=='__main__':unittest.main()
