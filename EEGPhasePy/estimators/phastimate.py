import numpy as np
import numpy.typing as npt
import scipy.signal as signal
from typing import Literal, Callable, Union
from statsmodels.regression.linear_model import yule_walker
import pygad
from bayes_opt import BayesianOptimization

from .estimator import Estimator
from ..utils.check import _check_array_dimensions, _check_type


class PHASTIMATE(Estimator):
    '''
    Class for the PHASTIMATE algorithm created by :cite:t:`Zrenner2020-zb`

    If you use this class, please cite :cite:t:`Zrenner2020-zb`

    PHASTIMATE uses an autoregressive approach to fill in data impacted by
    filter edge effects then applys a hilbert transform to extract the phase
    at the current time. It is best used in real-time environments where
    minimal delay in your system (particularly EEG -> computer) is guarenteed,
    because forward forecasting more than a couple of milliseconds beyond t = 0
    typically results in inaccurate phase predictions. This implementation does
    not support forward forecasting beyond t = 0

    PHASTIMATE also comes with genetic optimization for the phase estimation
    algorithms including order and window_edge. Our implementation of
    PHASTIMATE does not include optimization over filter parameters.
    '''

    def __init__(self, real_time_filter: npt.ArrayLike,
                 ground_truth_filter: npt.ArrayLike,
                 sampling_rate: int,
                 window_len=500,
                 window_edge=40,
                 ar_order=30):
        '''
        Constructor for PHASTIMATE class

        Parameters
        ----------
        real_time_filter
            Filter parameters for filter to apply for predicting phase.
            Accounts for FIR or IIR filters
        ground_truth_filter
            Filter parameters for identifying "true" phase. See
            :cite:t:Zrenner2020-zb for an in-depth discussion
            on "true" phase. Accounts for FIR or IIR filters
        sampling_rate : int
            Original sampling rate of data.
        window_len : int
            Window length in ms. Optional parameter to specify window length
            to run pseudo-real-time simulations or to train `window_len`
            dependent models with. This should match whatever is used in
            real-time
        window_edge : int
            Window edge to remove in ms. Optional parameter to specify edge to
            remove after applying `real_time_filter`
        ar_order : int
            The order for the auto-regressive model
        '''
        super().__init__(real_time_filter, ground_truth_filter,
                         sampling_rate, window_len, window_edge)

        _check_type(ar_order, ['int'])
        self.ar_order = ar_order

    def _ar_forecast(self, data: npt.ArrayLike,
                     ar_params: npt.ArrayLike, steps=10) -> np.ndarray:
        """
        Forecast future values from an AR process.

        Parameters
        ----------
        data : array-like
            Time series data.
        ar_params : array-like
            AR coefficients (phi_1, ..., phi_p).
        steps : int
            Number of steps to forecast.

        Returns
        --------
        forecasted_data : ndarray
            The forecasted series
        """
        p = len(ar_params)
        forecast = list(data[-p:])  # start with the last p values

        for _ in range(steps):
            new_val = np.dot(ar_params, forecast[-p:][::-1])  # weighted sum
            forecast.append(new_val)
            np.seterr(invalid='ignore')
            np.seterr(over='ignore')

        return np.array(forecast[p:])

    def _generate_black_box_function(self,
                                     data: npt.ArrayLike):
        '''
        Generates a function that runs a pseudo-real-time simulation of the
        autoregressive model on `data` and computes accuracy to peaks.
        The pseudo-real-time simulation will use the current window_len
        and will use a step of 0.01s. When the target phase is detected,
        the window will jump by 1 x its length

        Parameters
        -----------
        data : np.ndarray[float] | list[float]
            The (n_samples,) array containing the unfiltered EEG data to use
             for simulations

        Returns
        --------
        black_box_function : function
            The black box function that simulates the AR model with provided
            parameters and outputs accuracy
        '''
        _check_type(data, ['array'])
        _check_array_dimensions(data, [(1,)])

        def black_box_function(edge: float, ar_order: float) -> float:
            self.window_edge = int(edge)
            self.ar_order = int(ar_order)

            fs = self.sampling_rate
            window_i = 0
            window_len = self.window_len
            window_step = int(0.01*fs)

            triggers = []

            while window_i + window_len < len(data):
                window_data = data[window_i:window_i + window_len]
                if self.predict(window_data, 0, 10):
                    triggers.append(window_i + window_len)
                    window_i += window_len

                window_i += window_step

            accuracy = self.phase_accuracy_from_triggers(data, triggers, 0)

            return accuracy

        return black_box_function

    def _generate_fitness_function(self,
                                   data: npt.ArrayLike) \
            -> Callable[..., float]:
        '''
        Generates a function that runs a pseudo-real-time simulation of the
        autoregressive model on `data` and computes accuracy to peaks.
        The pseudo-real-time simulation will use provided window_len and will
        use a step of 0.05s. When the target phase is detected,
        the window will jump by 1 x its length

        Parameters
        -----------
        data : np.ndarray[float] | list[float]
            The (n_samples,) array containing the unfiltered EEG data to use
            for simulations

        Returns
        --------
        fitness_function : function
            The fitness function that simulates the AR model with provided
            parameters and outputs accuracy
        '''
        _check_type(data, ['array'])
        _check_array_dimensions(data, [(1,)])

        def fitness_function(ga_instance: pygad.GA,
                             solution: list[int],
                             solution_i: int) -> float:
            _solution = [int(sol) for sol in solution]

            self.window_edge = _solution[0]
            self.ar_order = _solution[1]

            fs = self.sampling_rate
            window_i = 0
            window_len = self.window_len
            window_step = int(0.05*fs)

            triggers = []

            while window_i + window_len < len(data):
                window_data = data[window_i:window_i + window_len]
                if self.predict(window_data, 0, 5):
                    triggers.append(window_i + window_len)
                    window_i += window_len

                window_i += window_step

            accuracy = self.phase_accuracy_from_triggers(data, triggers, 0)

            return accuracy

        return fitness_function

    def optimize_parameters(self,
                            data: npt.ArrayLike,
                            method: Literal["bayesian", "genetic"]
                            = "bayesian") -> None:
        '''
        Perform optimization over the amount of edge removed following
        filtering and autoregressive order. This method will update the
        properties of the current PHASTIMATE instance
        instance.

        Parameters
        -----------
        data : np.ndarray[float] | list[float]
            The (n_samples,) array containing the unfiltered EEG data to use
             for optimization
        method : "bayesian" | "genetic"
            Whether to perform bayesian optimization or genetic optimization
        '''
        if method == "bayesian":
            parameter_bounds = {
                "edge": [5, float(np.min([60, self.window_len / 8]))],
                "ar_order": [1.0, 0.1 * self.sampling_rate]
            }
            optimizer = BayesianOptimization(
                self._generate_black_box_function(data),
                pbounds=parameter_bounds)

            print(
                "[PHASTIMATE Bayesian Optimization] Starting bayesian" +
                " optimization....")
            optimizer.maximize(10, n_iter=100)
            print("[PHASTIMATE Bayesian Optimization] Optimization complete")

            print("[PHASTIMATE Bayesian Optimization] Accuracy of best " +
                  "parameters",
                  optimizer.max['target'])
            self.window_edge = int(optimizer.max["params"]["edge"])
            self.ar_order = int(optimizer.max["params"]["ar_order"])
        else:
            gene_space = [
                list(np.arange(10, np.min(
                    [80, self.window_len / 8]), 5, dtype=int)),
                list(np.arange(1, int(0.1 * self.sampling_rate), dtype=int))
            ]
            num_generations = 20
            num_parents_mating = 4

            fitness_function = self._generate_fitness_function(data)

            sol_per_pop = 5
            num_genes = len(gene_space)

            parent_selection_type = "rws"

            def on_gen(ga_instance) -> None:
                print("[PHASTIMATE Genetic Optimization] Generation : ",
                      ga_instance.generations_completed)
                print("[PHASTIMATE Genetic Optimization] Accuracy of the best"
                      + " solution :", ga_instance.best_solution()[1])

            print("[PHASTIMATE Genetic Optimization] Starting genetic " +
                  "optimization")
            ga_instance = pygad.GA(num_generations=num_generations,
                                   num_parents_mating=num_parents_mating,
                                   fitness_func=fitness_function,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_genes,
                                   gene_space=gene_space,
                                   parent_selection_type=parent_selection_type,
                                   on_generation=on_gen,
                                   mutation_percent_genes=50)
            ga_instance.run()

            solution, solution_fitness, solution_i = \
                ga_instance.best_solution()
            print("[PHASTIMATE Genetic Optimization] Optimization complete")
            print(
                "[PHASTIMATE Genetic Optimization]" +
                "Accuracy of best parameters: " + str(solution_fitness))

            self.window_edge = int(solution[0])
            self.ar_order = int(solution[1])

    def predict(self,
                data: npt.ArrayLike,
                target_phase: Union[float, int],
                tolerance: Union[float, int] = 5) -> bool:
        '''
        Predict whether the phase at the current time matches the target phase

        Parameters
        -----------
        data : np.ndarray[float] | list[float]
            The (n_samples,) array containing the unfiltered EEG data in the
            current window
        target_phase : float | int
            The phase in degrees that the current phase should match
        tolerance : float | int
            The tolerance between the current phase and target phase. The
            target phase has to be within `tolerance` degrees for this
            method to return `True`

        Returns
        --------
        current_phase_matches_target : bool
            Whether the current phase matches the target phase with a given
            tolerance

        '''
        _check_type(data, ['array'])
        _check_type(target_phase, ['int', 'float'])
        _check_type(tolerance, ['int', 'float'])
        _check_array_dimensions(data, [(1,)])

        edge = int((self.window_edge / 1000) * self.sampling_rate)
        filtered_data = self._filter_data(
            self.real_time_filter, data)[edge:-edge]

        ar_params, _ = yule_walker(filtered_data, self.ar_order)
        forecasted_data = self._ar_forecast(filtered_data, ar_params, 2*edge)
        full_data = np.concatenate([filtered_data, forecasted_data])

        analytic_signal = signal.hilbert(full_data)
        phase_t0 = np.angle(analytic_signal[-edge], deg=True) % 360

        if target_phase % 360 - tolerance < 0:
            return np.isclose(phase_t0, target_phase % 360, atol=tolerance) or\
                np.isclose(phase_t0, 360 - (target_phase %
                           360), atol=tolerance)
        else:
            return np.isclose(phase_t0, target_phase % 360, atol=tolerance)
