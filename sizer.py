import logging
import pickle
from datetime import datetime
from utils import add_years
import random
from pure_optimizer import PureOptimizer
from simulator_optimizer import SimulatorOptimizer
from copy import copy, deepcopy

class Sizer:
    def __init__(self, microgrid, time_series, sizing_config, initial_state=None):
        self.microgrid = microgrid
        self.time_series = time_series
        self.sizing_config = sizing_config
        self.initial_state = initial_state

    def size(self, selected_days, extracted_series, extracted_weights):
        if self.sizing_config.multi_stage_sizing:
            # Extend and apply progression
            selected_days, extracted_series, extracted_weights = self._extend_over_planning_horizon(selected_days,
                                                                                                    extracted_series,
                                                                                                    extracted_weights)
            # Check if NRD are enough --> compute reliability indicator
            # grid = self.microgrid.copy()
            # sc = deepcopy(self.sizing_config)
            # grid['parameters']['multi_stage_sizing'] = False
            # sc.multi_stage_sizing = False
            # optimizer_check_nrd = PureOptimizer(grid, selected_days, extracted_series,
            #                                extracted_weights, sc, self.time_series, use_365_days=False)
            # optimizer_check_nrd.optimize_function()
            # op_sizing_check_nrd = optimizer_check_nrd.optimal_sizing
            #
            # NPV_n_s = op_sizing_check_nrd['NPV']
            #
            # simu_optimizer_nrd = SimulatorOptimizer(grid, selected_days, extracted_series, extracted_weights,
            #                                         sc, self.time_series, op_sizing_check_nrd,
            #                                         use_365_days=True)
            # simu_optimizer_nrd.optimize_function()
            # NPV_365_s = simu_optimizer_nrd.optimal_sizing['NPV']
            #
            #
            # optimizer_check_nrd = PureOptimizer(grid, selected_days, extracted_series,
            #                                extracted_weights, sc, self.time_series, use_365_days=True)
            # optimizer_check_nrd.optimize_function()
            # op_sizing_check_nrd = optimizer_check_nrd.optimal_sizing
            #
            # NPV_365_s_star = op_sizing_check_nrd['NPV']
            #
            # reliability_nrd = 1
            # reliability_nrd -= abs(NPV_n_s - NPV_365_s)/NPV_365_s
            # reliability_nrd -= abs(NPV_365_s - NPV_365_s_star)/NPV_365_s_star
            # print("Reliability indicator NRD = %.3f"%reliability_nrd)


        else:
            # Run only on one year, apply progression to put ourself in a average situation for sizing.
            progressions = self._get_progressions()
            for device_name, progression in progressions.items():
                extracted_series[0][device_name] *= (1 + progression) ** (self.sizing_config.investment_horizon/2)
        #random.seed(42)

        self.optimizer = PureOptimizer(self.microgrid, selected_days, extracted_series, extracted_weights,
                                       self.sizing_config, self.time_series, use_365_days=False)
        self.optimal_grid = self.optimizer.optimize_function()  
        self.op_sizing = self.optimizer.optimal_sizing


    def _get_progressions(self):
        progressions = {}
        for device_type, devices in self.microgrid.items():
            # Note: it is a bit awkward to do this from the dictionary description, but we have no proper Grid object so far.
            for d in devices:
                try:
                    progressions[d['name']] = d['progression']
                except (KeyError, TypeError):
                    pass  # Skip if the attribute does not exist
        return progressions

    def _extend_over_planning_horizon(self, selected_days, extracted_series, extracted_weights):
        """

        :param selected_days: output of reprensentative days selection
        :param extracted_series: output of reprensentative days selection
        :param extracted_weights: output of reprensentative days selection
        :return: 3 lists with selected_days, extracted_series, and extracted_weights extended to the investment
        horizon with progressions applied if any.
        """
        logging.info('Extending representative days of first year over investment horizon with progressions...')

        # Simply replicate days and weights with a time shift of one year
        days = [{add_years(k, i): v for k, v in selected_days[0].items()} for i in range(self.sizing_config.investment_horizon)]
        weights = [{add_years(k, i): v for k, v in extracted_weights[0].items()} for i in range(self.sizing_config.investment_horizon)]

        # Extend series to investment horizon
        init_series = extracted_series[0].copy()
        series = [init_series]
        current_series = init_series.copy()
        # Fetch progressions first, if any.
        progressions = self._get_progressions()

        for i in range(1, self.sizing_config.investment_horizon):
            # Compute number of days of the year (some have 366 ...)
            current_year = current_series.index[0].year
            one_year_delta = (datetime(current_year + 1, 1, 1) - datetime(current_year, 1, 1))

            new_series = current_series.copy()
            # Add a year
            new_series.index += one_year_delta
            # apply progressions
            for device_name, progression in progressions.items():
                new_series[device_name] *= (1 + progression)

            series.append(new_series)
            current_series = new_series

        return days, series, weights