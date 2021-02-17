# -*- coding: UTF-8 -*-
"""
Management of timeseries of the sizing tool.
"""

from io import StringIO
import datetime
import csv
import logging
import pickle
import daysxtractor
import os
import shutil
from daysxtractor import Bins

def create_rd_output_path(output_path):
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)
    return output_path

class RepresentativeDays:
    def __init__(self):
        pass
    
    def get_representative_days(self, time_series, sizing_config, output_path=""):
        """
        Extract a sample of days appropriately weighted that is able to represent the full set of available days.

        :return: Dictionaries with the selected days and weights, reduced timeseries, list of the weights for each corresponding day.
        """

        sc = sizing_config

        # Select/Collect representative days
        logging.info("Select %s representative days..." % sc.num_representative_days)
        try:
            with open(sc.save_days, 'rb') as f:
                selected_days, extracted_series, extracted_weights = pickle.load(f)
                logging.info('\t reusing representative days from %s. ' % sc.save_days)
        except (FileNotFoundError, ValueError) as e:
            # logging.warning('Error loading representative days from "%s": %s.' % (sc.save_days, e))

            # Get representative days as a dict with the selected days and weights.
            selected_days = self.representative_days(time_series, n=sc.num_representative_days,
                                                timelimit=sc.representative_days_time_limit,
                                                solver=sc.representative_days_solver, output_path=output_path)

        
            logging.info(selected_days)

            # Extract time series
            extracted_series, extracted_weights = self.extract_days(time_series, selected_days)
            selected_days = [selected_days]
            extracted_series = [extracted_series]
            extracted_weights = [extracted_weights]

            if sc.save_days is not '':
                with open(sc.save_days, 'wb') as f:
                    pickle.dump([selected_days, extracted_series, extracted_weights], f)

        return selected_days, extracted_series, extracted_weights


    def representative_days(self, timeseries, output_path, n=24, solver=None, timelimit=60.0):
        """
        Get representative days from a time series.

        :param output_path: Path were plots are stored.
        :param timeseries: Pandas DataFrame with timestamp as index.
        :param n: Number of representative days to select.
        :param solver: Solver name.
        :param timelimit: Time limit [seconds.]
        :return: Dictionary with the representative days and the weights. The sum of the weights is equal to the number of days in the time series.
        """

        # Dump the timeseries in CSV.
        csv = StringIO()
        timeseries.to_csv(csv)
        csv.seek(0)

        # Launch daysxtractor
        xtractor_data = daysxtractor.parseData(csv)
        if solver is None:
            daySelector = daysxtractor.SamplingDaysSelector(numberRepresentativeDays=n, timelimit=timelimit, verbose=True)
        else:
            daySelector = daysxtractor.MIPDaysSelector(numberRepresentativeDays=n, timelimit=timelimit, solverName=solver,
                                                    verbose=True)
        days = daySelector.selectDays(xtractor_data)

        # Error measures
        bins = Bins(xtractor_data, daySelector.binsPerTimeSeries)
        representativeBins = Bins()
        representativeBins.createFromRepresentativeDays(bins, days)

        logging.info("\nError measures:")
        logging.info("\t- Bins population:")
        for p in bins.labelRanges():
            populationMin, populationMax = bins.population(p)
            logging.info(
                "\t\t%s: min=%.2f%%, max=%.2f%%" % (bins.labels[p].name, populationMin * 100.0, populationMax * 100.0))

        logging.info("\t- Normalized root-mean-square error:")
        for p in bins.labelRanges():
            logging.info("\t\t%s: %.2f%%" % (bins.labels[p].name, bins.nrmsError(p, representativeBins) * 100.0))

        logging.info("\t- Relative area error:")
        for p in bins.labelRanges():
            logging.info("\t\t%s: %.2f%%" % (bins.labels[p].name, bins.relativeAreaError(p, representativeBins) * 100.0))

        # Plots
        for label in xtractor_data.labels:
            xtractor_data.plotRepresentativeTimeseries(label, days, pathPrefix=output_path)

        for p in bins.labelRanges():
            if bins.labels[p].name == "Conso":
                LD_nrmse = bins.nrmsError(p, representativeBins)

        return days


    def extract_days(self, df, days):
        """
        Extract selected days from a complete time series.

        :param df: Pandas DataFrame.
        :param days: Dictionary of representative days.
        :return: Extracted time series as a pandas DataFrame, weights where keys are readapted.
        """
        # Extract
        days_list = sorted(list(days.keys()))
        days_list.append(days_list[-1] + datetime.timedelta(days=1))  # Add one day fro computation needs
        print(days_list)
        extracted = df[df.index.map(lambda x: x.to_pydatetime().date() in days_list)]

        # Reindex
        t0 = df.iloc[0].name
        dt = df.iloc[1].name - t0
        extracted.index = [t0 + i * dt for i in range(len(extracted))]

        # Weights
        d0 = t0.to_pydatetime().date()
        weights = {d0 + datetime.timedelta(days=i): days[d] for i, d in enumerate(days_list[:-1])}

        return extracted, weights


    def write_days(self, days, path):
        """
        Write weighted days into a CSV file.

        :param days: Dictionary with days and weights.
        :param path: Output path.
        """
        with open(path, 'w', newline='') as file:
            csv_file = csv.writer(file)
            csv_file.writerow(['Day', 'Weight'])
            for d in sorted(days.keys()):
                csv_file.writerow([d, days[d]])
