
import math
import jax
import optax
import matplotlib.pyplot as plt
from functools import partial
from imnn.utils.utils import _check_boolean, _check_type, \
    _check_model, _check_model_output, _check_optimiser, _check_state, \
    _check_statistics_set
from imnn.experimental import progress_bar

import jax.numpy as np

from progress_bar import *
from imnns import *

#from imnn.utils import value_and_jacrev
#from imnn.utils.utils import _check_simulator


#@title imnn utils <font color='lightgreen'>[run me]</font>
def _check_input(input, shape, name, allow_None=False):
    """Exception raising checks for numpy array shapes

    Checks whether input is not ``None`` and if not checks that the input is a
    jax numpy array and if not raises a warning. If the input is a jax numpy
    array it then checks the shape is the same as the required shape.

    Can also allow ``None`` to be passed if it input is not essential.

    Parameters
    ----------
    input: any
        The input parameter to be checked
    shape: tuple
        The shape that the input is required to be
    name: str
        The name of the variable for printing explicit errors in ``Exception``
    allow_None: bool, default=False
        Whether a ``None`` input can be returned as None without raising error

    Returns
    -------
    array
        Returns the input if all checks pass

    Raises
    ------
    ValueError
        If input is None
    ValueError
        If input shape is incorrect
    TypeError
        If input is not a jax array
    """
    if (input is None) and (not allow_None):
        raise ValueError(f"`{name}` is None")
    elif (input is None) and allow_None:
        return input
    # elif not isinstance(
    #         input, (jax.interpreters.xla.device_array, np.ndarray)):
    #     raise TypeError(f"`{name}` must be a jax array")
    else:
        if input.shape != shape:
            raise ValueError(f"`{name}` should have shape {shape} but has " +
                             f"{input.shape}")
    return input



class _SGD_IMNN:
    """Information maximising neural network parent class

    This class defines the general fitting framework for information maximising
    neural networks. It includes the generic calculations of the Fisher
    information matrix from the outputs of a neural network as well as an XLA
    compilable fitting routine (with and without a progress bar). This class
    also provides a plotting routine for fitting history and a function to
    calculate the score compression of network outputs to quasi-maximum
    likelihood estimates of model parameter values.

    The outline of the fitting procedure is that a set of :math:`i\\in[1, n_s]`
    simulations and :math:`n_d` derivatives with respect to physical model
    parameters are used to calculate network outputs and their derivatives
    with respect to the physical model parameters, :math:`{\\bf x}^i` and
    :math:`\\partial{{\\bf x}^i}/\\partial\\theta_\\alpha`, where
    :math:`\\alpha` labels the physical parameter. The exact details of how
    these are calculated depend on the type of available data (see list of
    different IMNN below). With :math:`{\\bf x}^i` and
    :math:`\\partial{{\\bf x}^i}/\\partial\\theta_\\alpha` the covariance

    .. math::
        C_{ab} = \\frac{1}{n_s-1}\\sum_{i=1}^{n_s}(x^i_a-\\mu^i_a)
        (x^i_b-\\mu^i_b)

    and the derivative of the mean of the network outputs with respect to the
    model parameters

    .. math::
        \\frac{\\partial\\mu_a}{\\partial\\theta_\\alpha} = \\frac{1}{n_d}
        \\sum_{i=1}^{n_d}\\frac{\\partial{x^i_a}}{\\partial\\theta_\\alpha}

    can be calculated and used form the Fisher information matrix

    .. math::
        F_{\\alpha\\beta} = \\frac{\\partial\\mu_a}{\\partial\\theta_\\alpha}
        C^{-1}_{ab}\\frac{\\partial\\mu_b}{\\partial\\theta_\\beta}.

    The loss function is then defined as

    .. math::
        \\Lambda = -\\log|{\\bf F}| + r(\\Lambda_2) \\Lambda_2

    Since any linear rescaling of a sufficient statistic is also a sufficient
    statistic the negative logarithm of the determinant of the Fisher
    information matrix needs to be regularised to fix the scale of the network
    outputs. We choose to fix this scale by constraining the covariance of
    network outputs as

    .. math::
        \\Lambda_2 = ||{\\bf C}-{\\bf I}|| + ||{\\bf C}^{-1}-{\\bf I}||

    Choosing this constraint is that it forces the covariance to be
    approximately parameter independent which justifies choosing the covariance
    independent Gaussian Fisher information as above. To avoid having a dual
    optimisation objective, we use a smooth and dynamic regularisation strength
    which turns off the regularisation to focus on maximising the Fisher
    information when the covariance has set the scale

    .. math::
        r(\\Lambda_2) = \\frac{\\lambda\\Lambda_2}{\\Lambda_2-\\exp
        (-\\alpha\\Lambda_2)}.

    Once the loss function is calculated the automatic gradient is then
    calculated and used to update the network parameters via the optimiser
    function. Note for large input data-sizes, large :math:`n_s` or massive
    networks the gradients may need manually accumulating via the
    :func:`~imnn.imnn._aggregated_imnn._AggregatedIMNN`.

    ``_IMNN`` is designed as the parent class for a range of specific case
    IMNNs. There is a helper function (IMNN) which should return the correct
    case when provided with the correct data. These different subclasses are:

    :func:`~imnn.SimulatorIMNN`:

        Fit an IMNN using simulations generated on-the-fly from a jax (XLA
        compilable) simulator

    :func:`~imnn.GradientIMNN`:

        Fit an IMNN using a precalculated set of fiducial simulations and their
        derivatives with respect to model parameters

    :func:`~imnn.NumericalGradientIMNN`:

        Fit an IMNN using a precalculated set of fiducial simulations and
        simulations generated using parameter values just above and below the
        fiducial parameter values to make a numerical estimate of the
        derivatives of the network outputs. Best stability is achieved when
        seeds of the simulations are matched between all parameter directions
        for the numerical derivative

    :func:`~imnn.AggregatedSimulatorIMNN`:

        ``SimulatorIMNN`` distributed over multiple jax devices and gradients
        aggregated manually. This might be necessary for very large input sizes
        as batching cannot be done when calculating the Fisher information
        matrix

    :func:`~imnn.AggregatedGradientIMNN`:

        ``GradientIMNN`` distributed over multiple jax devices and gradients
        aggregated manually. This might be necessary for very large input sizes
        as batching cannot be done when calculating the Fisher information
        matrix

    :func:`~imnn.AggregatedNumericalGradientIMNN`:

        ``NumericalGradientIMNN`` distributed over multiple jax devices and
        gradients aggregated manually. This might be necessary for very large
        input sizes as batching cannot be done when calculating the Fisher
        information matrix

    :func:`~imnn.DatasetGradientIMNN`:

        ``AggregatedGradientIMNN`` with prebuilt TensorFlow datasets

    :func:`~imnn.DatasetNumericalGradientIMNN`:

        ``AggregatedNumericalGradientIMNN`` with prebuilt TensorFlow datasets

    There are currently two other parent classes

    :func:`~imnn.imnn._aggregated_imnn.AggregatedIMNN`:

        This is the parent class which provides the fitting routine when the
        gradients of the network parameters are aggregated manually rather than
        automatically by jax. This is necessary if the size of an entire batch
        of simulations (and their derivatives with respect to model parameters)
        and the network parameters and their calculated gradients is too large
        to fit into memory. Note there is a significant performance loss from
        using the aggregation so it should only be used for these large data
        cases

    Parameters
    ----------
    n_s : int
        Number of simulations used to calculate network output covariance
    n_d : int
        Number of simulations used to calculate mean of network output
        derivative with respect to the model parameters
    n_params : int
        Number of model parameters
    n_summaries : int
        Number of summaries, i.e. outputs of the network
    input_shape : tuple
        The shape of a single input to the network
    θ_fid : float(n_params,)
        The value of the fiducial parameter values used to generate inputs
    validate : bool
        Whether a validation set is being used
    simulate : bool
        Whether input simulations are generated on the fly
    _run_with_pbar : bool
        Book keeping parameter noting that a progress bar is used when
        fitting (induces a performance hit). If ``run_with_pbar = True``
        and ``run_without_pbar = True`` then a jit compilation error will
        occur and so it is prevented
    _run_without_pbar : bool
        Book keeping parameter noting that a progress bar is not used when
        fitting. If ``run_with_pbar = True`` and ``run_without_pbar = True``
        then a jit compilation error will occur and so it is prevented
    F : float(n_params, n_params)
        Fisher information matrix calculated from the network outputs
    invF : float(n_params, n_params)
        Inverse Fisher information matrix calculated from the network outputs
    C : float(n_summaries, n_summaries)
        Covariance of the network outputs
    invC : float(n_summaries, n_summaries)
        Inverse covariance of the network outputs
    μ : float(n_summaries,)
        Mean of the network outputs
    dμ_dθ : float(n_summaries, n_params)
        Derivative of the mean of the network outputs with respect to model
        parameters
    state : :obj:state
        The optimiser state used for updating the network parameters and
        optimisation algorithm
    initial_w : list
        List of the network parameters values at initialisation (to restart)
    final_w : list
        List of the network parameters values at the end of fitting
    best_w : list
        List of the network parameters values which provide the maxmimum value
        of the determinant of the Fisher matrix
    w : list
        List of the network parameters values (either final or best depending
        on setting when calling fit(...))
    history : dict
        A dictionary containing the fitting history. Keys are
            - **detF** -- determinant of the Fisher information at the end of
              each iteration
            - **detC** -- determinant of the covariance of network outputs at
              the end of each iteration
            - **detinvC** -- determinant of the inverse covariance of network
              outputs at the end of each iteration
            - **Λ2** -- value of the covariance regularisation at the end of
              each iteration
            - **r** -- value of the regularisation coupling at the end of each
              iteration
            - **val_detF** -- determinant of the Fisher information of the
              validation data at the end of each iteration
            - **val_detC** -- determinant of the covariance of network outputs
              given the validation data at the end of each iteration
            - **val_detinvC** -- determinant of the inverse covariance of
              network outputs given the validation data at the end of each
              iteration
            - **val_Λ2** -- value of the covariance regularisation given the
              validation data at the end of each iteration
            - **val_r** -- value of the regularisation coupling given the
              validation data at the end of each iteration
            - **max_detF** -- maximum value of the determinant of the Fisher
              information on the validation data (if available)

    Methods
    -------
    model:
        Neural network as a function of network parameters and inputs
    _get_parameters:
        Function which extracts the network parameters from the state
    _model_initialiser:
        Function to initialise neural network weights from RNG and shape tuple
    _opt_initialiser:
        Function which generates the optimiser state from network parameters
    _update:
        Function which updates the state from a gradient

    Todo
    ----
    - Finish all docstrings and documentation
    - Update `NoiseNumericalGradientIMNN` to inherit from `_AggregatedIMNN`

    """
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid,
                 model, optimiser, key_or_state, dummy_graph_input=None, no_invC=False, do_reg=True,
                 evidence=False):
        """Constructor method

        Initialises all _IMNN attributes, constructs neural network and its
        initial parameter values and creates history dictionary

        Parameters
        ----------
        n_s : int
            Number of simulations used to calculate summary covariance
        n_d : int
            Number of simulations used to calculate mean of summary derivative
        n_params : int
            Number of model parameters
        n_summaries : int
            Number of summaries, i.e. outputs of the network
        input_shape : tuple
            The shape of a single input to the network
        θ_fid : float(n_params,)
            The value of the fiducial parameter values used to generate inputs
        model : tuple, len=2
            Tuple containing functions to initialise neural network
            ``fn(rng: int(2), input_shape: tuple) -> tuple, list`` and the
            neural network as a function of network parameters and inputs
            ``fn(w: list, d: float([None], input_shape)) -> float([None],
            n_summaries)``.
            (Essentibly stax-like, see `jax.experimental.stax <https://jax.read
            thedocs.io/en/stable/jax.experimental.stax.html>`_))
        optimiser : tuple, len=3
            Tuple containing functions to generate the optimiser state
            ``fn(x0: list) -> :obj:state``, to update the state from a list of
            gradients ``fn(i: int, g: list, state: :obj:state) -> :obj:state``
            and to extract network parameters from the state
            ``fn(state: :obj:state) -> list``.
            (See `jax.experimental.optimizers <https://jax.readthedocs.io/en/st
            able/jax.experimental.optimizers.html>`_)
        key_or_state : int(2) or :obj:state
            Either a stateless random number generator or the state object of
            an preinitialised optimiser
        dummy_graph_input : jraph.GraphsTuple or 'jax.numpy.DeviceArray'
            Either a (padded) graph input or device array. If supplied ignores 
            `input_shape` parameter
        """
        self.dummy_graph_input=dummy_graph_input
        self._initialise_parameters(
            n_s, n_d, n_params, n_summaries, input_shape, θ_fid)
        self._initialise_model(model, optimiser, key_or_state)
        self._initialise_history()
        self.no_invC=no_invC
        self.do_reg=do_reg
        self.evidence=evidence


    def _initialise_parameters(self, n_s, n_d, n_params, n_summaries,
                               input_shape, θ_fid):
        """Performs type checking and initialisation of class attributes

        Parameters
        ----------
        n_s : int
            Number of simulations used to calculate summary covariance
        n_d : int
            Number of simulations used to calculate mean of summary derivative
        n_params : int
            Number of model parameters
        n_summaries : int
            Number of summaries, i.e. outputs of the network
        input_shape : tuple
            The shape of a single input to the network
        θ_fid : float(n_params,)
            The value of the fiducial parameter values used to generate inputs

        Raises
        ------
        TypeError
            Any of the parameters are not correct type
        ValueError
            Any of the parameters are ``None``
            ``Θ_fid`` has the wrong shape
        """
        self.n_s = n_s #_check_type(n_s, int, "n_s")
        self.n_d = n_d #_check_type(n_d, int, "n_d")
        self.n_params = n_params #_check_type(n_params, int, "n_params")
        self.n_summaries = n_summaries #_check_type(n_summaries, int, "n_summaries")
        self.input_shape = input_shape #_check_type(input_shape, tuple, "input_shape")
        self.θ_fid = θ_fid #_check_input(θ_fid, (self.n_params,), "θ_fid")

        self.validate = False
        self.simulate = False
        self._run_with_pbar = False
        self._run_without_pbar = False

        self.F = None
        self.invF = None
        self.C = None
        self.invC = None
        self.μ = None
        self.dμ_dθ = None

        self._model_initialiser = None
        self.model = None
        self._opt_initialiser = None
        self._update = None
        self._get_parameters = None
        self.state = None
        self.initial_w = None
        self.final_w = None
        self.best_w = None
        self.w = None

        self.history = None

    def _initialise_model(self, model, optimiser, key_or_state):
        """Initialises neural network parameters or loads optimiser state

        Parameters
        ----------
        model : tuple, len=2
            Tuple containing functions to initialise neural network
            ``fn(rng: int(2), input_shape: tuple) -> tuple, list`` and
            the neural network as a function of network parameters and inputs
            ``fn(w: list, d: float([None], input_shape)) -> float([None],
            n_summaries)``. (Essentibly stax-like, see `jax.experimental.stax
            <https://jax.readthedocs.io/en/stable/jax.experimental.stax.html>`_
            ))
        optimiser : tuple or obj, len=3
            Tuple containing functions to generate the optimiser state
            ``fn(x0: list) -> :obj:state``, to update the state from a list of
            gradients ``fn(i: int, g: list, state: :obj:state) -> :obj:state``
            and to extract network parameters from the state
            ``fn(state: :obj:state) -> list``.
            (See `jax.experimental.optimizers <https://jax.readthedocs.io/en/st
            able/jax.experimental.optimizers.html>`_)
        key_or_state : int(2) or :obj:state
            Either a stateless random number generator or the state object of
            an preinitialised optimiser

        Notes
        -----
        The design of the model follows `jax's stax module <https://jax.readth
        edocs.io/en/latest/jax.experimental.stax.html>`_ in that the model is
        encapsulated by two functions, one to initialise the network and one to
        call the model, i.e.::

            import jax
            from jax.experimental import stax

            rng = jax.random.PRNGKey(0)

            data_key, model_key = jax.random.split(rng)

            input_shape = (10,)
            inputs = jax.random.normal(data_key, shape=input_shape)

            model = stax.serial(
                stax.Dense(10),
                stax.LeakyRelu,
                stax.Dense(10),
                stax.LeakyRelu,
                stax.Dense(2))

            output_shape, initial_params = model[0](model_key, input_shape)

            outputs = model[1](initial_params, inputs)

        Note that the model used in the IMNN is assumed to be totally
        broadcastable, i.e. any batch shape can be used for inputs. This might
        require having a layer which reshapes all batch dimensions into a
        single dimension and then unwraps it at the last layer. A model such as
        that above is already fully broadcastable.

        The optimiser should follow `jax's experimental optimiser module <http
        s://jax.readthedocs.io/en/stable/jax.experimental.optimizers.html>`_ in
        that the optimiser is encapsulated by three functions, one to
        initialise the state, one to update the state from a list of gradients
        and one to extract the network parameters from the state, .i.e

        .. code-block:: python

            from jax.experimental import optimizers
            import jax.numpy as np

            optimiser = optimizers.adam(step_size=1e-3)

            initial_state = optimiser[0](initial_params)
            params = optimiser[2](initial_state)

            def scalar_output(params, inputs):
                return np.sum(model[1](params, inputs))

            counter = 0
            grad = jax.grad(scalar_output, argnums=0)(params, inputs)
            state = optimiser[1](counter, grad, state)

        This function either initialises the neural network or the state if
        passed a stateless random number generator in ``key_or_state`` or loads
        a predefined state if the state is passed to ``key_or_state``. The
        functions get mapped to the class functions

        .. code-block:: python

            self.model = model[1]
            self._model_initialiser = model[0]

            self._opt_initialiser = optimiser[0]
            self._update = optimiser[1]
            self._get_parameters = optimiser[2]

        The state is made into the ``state`` class attribute and the parameters
        are assigned to ``initial_w``, ``final_w``, ``best_w`` and ``w`` class
        attributes (where ``w`` stands for weights).

        There is some type checking done, but for freedom of choice of model
        there will be very few raised warnings.

        Raises
        ------
        TypeError
            If the random number generator is not correct, or if there is no
            possible way to construct a model or an optimiser from the passed
            parameters
        ValueError
            If any input is ``None`` or if the functions for the model or
            optimiser do not conform to the necessary specifications
        """

        # initialize FLAX model here
        self._model_initialiser = model.init
        self.model = model.apply

        # unpack optimiser
        self._opt_initialiser, self._update = optimiser

        #state, key = _check_state(key_or_state)
        key = key_or_state

        if key is not None:
            key = _check_input(key, (2,), "key_or_state")
            if self.dummy_graph_input is None:
                dummy_x = jax.random.uniform(key, self.input_shape)
            else:
                dummy_x = self.dummy_graph_input

            # INITIAL PARAMS
            self.initial_w = self._model_initialiser(key, dummy_x)
            
            # DUMMY OUTPUT
            output = self.model(self.initial_w, dummy_x)
            # check to see if right shape
            #_check_model_output(output.shape, (self.n_summaries,))
            # INITIAL STATE
            self.state = self._opt_initialiser(self.initial_w)


        else:
            self.state = state
            try:
                self._get_parameters(self.state)
            except Exception:
                raise TypeError("`state` is not valid for extracting " +
                                "parameters from")

        self.dummy_x = dummy_x
        self.initial_w = self._model_initialiser(key, dummy_x)
        self.final_w = self._model_initialiser(key, dummy_x)
        self.best_w = self._model_initialiser(key, dummy_x)
        self.w = self._model_initialiser(key, dummy_x)


    def _initialise_history(self):
        """Initialises history dictionary attribute

        Notes
        -----
        The contents of the history dictionary are
            - **detF** -- determinant of the Fisher information at the end of
              each iteration
            - **detC** -- determinant of the covariance of network outputs at
              the end of each iteration
            - **detinvC** -- determinant of the inverse covariance of network
              outputs at the end of each iteration
            - **Λ2** -- value of the covariance regularisation at the end of
              each iteration
            - **r** -- value of the regularisation coupling at the end of each
              iteration
            - **val_detF** -- determinant of the Fisher information of the
              validation data at the end of each iteration
            - **val_detC** -- determinant of the covariance of network outputs
              given the validation data at the end of each iteration
            - **val_detinvC** -- determinant of the inverse covariance of
              network outputs given the validation data at the end of each
              iteration
            - **val_Λ2** -- value of the covariance regularisation given the
              validation data at the end of each iteration
            - **val_r** -- value of the regularisation coupling given the
              validation data at the end of each iteration
            - **max_detF** -- maximum value of the determinant of the Fisher
              information on the validation data (if available)

        """
        self.history = {
            "detF": np.zeros((0,)),
            "detC": np.zeros((0,)),
            "detinvC": np.zeros((0,)),
            "Λ2": np.zeros((0,)),
            "r": np.zeros((0,)),
            "val_detF": np.zeros((0,)),
            "val_detC": np.zeros((0,)),
            "val_detinvC": np.zeros((0,)),
            "val_Λ2": np.zeros((0,)),
            "val_r": np.zeros((0,)),
            "max_detF": np.float32(0.)
        }

    def _set_history(self, results):
        """Places results from fitting into the history dictionary

        Parameters
        ----------
        results : list
            List of results from fitting procedure. These are:
                - **detF** *(float(n_iterations, 2))* -- determinant of the
                  Fisher information, ``detF[:, 0]`` for training and
                  ``detF[:, 1]`` for validation
                - **detC** *(float(n_iterations, 2))* -- determinant of the
                  covariance of network outputs, ``detC[:, 0]`` for training
                  and ``detC[:, 1]`` for validation
                - **detinvC** *(float(n_iterations, 2))* -- determinant of the
                  inverse covariance of network outputs, ``detinvC[:, 0]`` for
                  training and ``detinvC[:, 1]`` for validation
                - **Λ2** *(float(n_iterations, 2))* -- value of the covariance
                  regularisation, ``Λ2[:, 0]`` for training and ``Λ2[:, 1]``
                  for validation
                - **r** *(float(n_iterations, 2))* -- value of the
                  regularisation coupling, ``r[:, 0]`` for training and
                  ``r[:, 1]`` for validation

        """
        keys = ["detF", "detC", "detinvC", "Λ2", "r"]
        for result, key in zip(results, keys):
            self.history[key] = np.hstack([self.history[key], result[:, 0]])
            if self.validate:
                self.history[f"val_{key}"] = np.hstack(
                    [self.history[f"val_{key}"], result[:, 1]])

    def _set_inputs(self, rng, max_iterations):
        """Builds list of inputs for the XLA compilable fitting routine

        Parameters
        ----------
        rng : int(2,) or None
            A stateless random number generator
        max_iterations : int
            Maximum number of iterations to run the fitting procedure for

        Notes
        -----
        The list of inputs to the routine are
            - **max_detF** *(float)* -- The maximum value of the determinant of
              the Fisher information matrix calculated so far. This is zero if
              not run before or the value from previous calls to ``fit``
            - **best_w** *(list)* -- The value of the network parameters which
              obtained the maxmimum determinant of the Fisher information
              matrix. This is the initial network parameter values if not run
              before
            - **detF** *(float(max_iterations, 1) or
              float(max_iterations, 2))* -- A container for all possible values
              of the determinant of the Fisher information matrix during each
              iteration of fitting. If there is no validation (for simulation
              on-the-fly for example) then this container has a shape of
              ``(max_iterations, 1)``, otherwise validation values are stored
              in ``detF[:, 1]``.
            - **detC** *(float(max_iterations, 1) or
              float(max_iterations, 2))* -- A container for all possible values
              of the determinant of the covariance of network outputs during
              each iteration of fitting. If there is no validation (for
              simulation on-the-fly for example) then this container has a
              shape of ``(max_iterations, 1)``, otherwise validation values are
              stored in ``detC[:, 1]``.
            - **detF** *(float(max_iterations, 1) or
              float(max_iterations, 2))* -- A container for all possible values
              of the determinant of the inverse covariance of network outputs
              during each iteration of fitting. If there is no validation (for
              simulation on-the-fly for example) then this container has a
              shape of ``(max_iterations, 1)``, otherwise validation values are
              stored in ``detinvC[:, 1]``.
            - **Λ2** *(float(max_iterations, 1) or
              float(max_iterations, 2))* -- A container for all possible values
              of the covariance regularisation during each iteration of
              fitting. If there is no validation (for simulation on-the-fly for
              example) then this container has a shape of
              ``(max_iterations, 1)``, otherwise validation values are stored
              in ``Λ2[:, 1]``.
            - **r** *(float(max_iterations, 1) or
              float(max_iterations, 2))* -- A container for all possible values
              of the regularisation coupling strength during each iteration of
              fitting. If there is no validation (for simulation on-the-fly for
              example) then this container has a shape of
              ``(max_iterations, 1),`` otherwise validation values are stored
              in ``r[:, 1]``.
            - **counter** *(int)* -- Iteration counter used to note whether the
              while loop reaches ``max_iterations``. If not, the history
              objects (above) get truncated to length ``counter``. This starts
              with value zero
            - **patience_counter** *(int)* -- Counts the number of iterations
              where there is no increase in the value of the determinant of the
              Fisher information matrix, used for early stopping. This starts
              with value zero
            - **state** *(:obj:state)* -- The current optimiser state used for
              updating the network parameters and optimisation algorithm
            - **rng** *(int(2,))* -- A stateless random number generator which
              gets updated on each iteration

        Todo
        ----
        ``rng`` is currently only used for on-the-fly simulation but could
        easily be updated to allow for stochastic models
        """
        if self.validate:
            shape = (max_iterations, 2)
        else:
            shape = (max_iterations, 1)

        return (
            self.history["max_detF"], self.best_w, np.zeros(shape),
            np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape),
            np.int32(0), np.int32(0), self.state, self.w, rng)

    def fit(self, λ, ϵ, γ=1000., rng=None, patience=100, min_iterations=100,
            max_iterations=int(1e5), print_rate=None, best=True):


        @jax.jit
        def _fit(inputs):

            return jax.lax.while_loop(
                partial(self._fit_cond, patience=patience,
                        max_iterations=max_iterations),
                partial(self._fit, λ=λ, α=α, γ=γ, min_iterations=min_iterations),
                inputs)

        def _fit_pbar(inputs):

            return jax.lax.while_loop(
                progress_bar(max_iterations, patience, print_rate)(
                    partial(self._fit_cond, patience=patience,
                            max_iterations=max_iterations)),
                jax.jit(
                    partial(self._fit, λ=λ, α=α,
                            min_iterations=min_iterations)),
                inputs)

        λ = λ #_check_type(λ, float, "λ")
        ϵ = ϵ #_check_type(ϵ, float, "ϵ")
        γ = γ #_check_type(γ, float, "γ")
        α = self.get_α(λ, ϵ)
        patience = patience #_check_type(patience, int, "patience")
        min_iterations = min_iterations #_check_type(min_iterations, int, "min_iterations")
        max_iterations = max_iterations #_check_type(max_iterations, int, "max_iterations")
        best = best #_check_boolean(best, "best")
        if self.simulate and (rng is None):
            raise ValueError("`rng` is necessary when simulating.")
        rng = _check_input(rng, (2,), "rng", allow_None=True)
        inputs = self._set_inputs(rng, max_iterations)
        if print_rate is None:
            if self._run_with_pbar:
                raise ValueError(
                    "Cannot run IMNN without progress bar after running " +
                    "with progress bar. Either set `print_rate` to an int " +
                    "or reinitialise the IMNN.")
            else:
                self._run_without_pbar = True
                results = _fit(inputs)
        else:
            if self._run_without_pbar:
                raise ValueError(
                    "Cannot run IMNN with progress bar after running " +
                    "without progress bar. Either set `print_rate` to None " +
                    "or reinitialise the IMNN.")
            else:
                print_rate = _check_type(print_rate, int, "print_rate")
                self._run_with_pbar = True
                results = _fit_pbar(inputs)
        self.history["max_detF"] = results[0]
        self.best_w = results[1]
        self._set_history(
            (results[2][:results[7]],
             results[3][:results[7]],
             results[4][:results[7]],
             results[5][:results[7]],
             results[6][:results[7]]))
        if len(results) == 12:
            self.state = results[-3]
        self.final_w = results[-2] #self._get_parameters(self.state)
        if best:
            w = self.best_w
        else:
            w = self.final_w
        self.set_F_statistics(w, key=rng)

    def _get_fitting_keys(self, rng):
        """Generates random numbers for simulation generation if needed

        Parameters
        ----------
        rng : int(2,) or None
            A random number generator

        Returns
        -------
        int(2,), int(2,), int(2,) or None, None, None:
            A new random number generator and random number generators for
            training and validation, or empty values
        """
        if rng is not None:
            return jax.random.split(rng, num=3)
        else:
            return None, None, None

    def get_α(self, λ, ϵ):
        """Calculate rate parameter for regularisation from closeness criterion

        Parameters
        ----------
        λ : float
            coupling strength of the regularisation
        ϵ : float
            closeness criterion describing how close to the 1 the determinant
            of the covariance (and inverse covariance) of the network outputs
            is desired to be

        Returns
        -------
        float:
            The steepness of the tanh-like function (or rate) which determines
            how fast the determinant of the covariance of the network outputs
            should be pushed to 1
        """
        return - math.log(ϵ * (λ - 1.) + ϵ ** 2. / (1 + ϵ)) / ϵ

    def _fit(self, inputs, λ=None, α=None, γ=None,  min_iterations=None):
        

        max_detF, best_w, detF, detC, detinvC, Λ2, r, \
            counter, patience_counter, state, w, rng = inputs
        rng, training_key, validation_key = self._get_fitting_keys(rng)

        # different pieces of the gradient
        grad, results = jax.grad(
            self._get_loss_reg, argnums=0, has_aux=True)(w, λ, α, γ, training_key)
        dC_dw = self._calculate_Cw(w, training_key)
        d_dμdθ_w = jax.jacrev(
            self._get_loss_dervs, argnums=0)(w, λ, α, γ, training_key)

        _F, _C, _invC, _Λ2, _r, _dμ_dθ = results

        results = results[:-1]
        _invF = np.linalg.inv(_F)

        # GET DIFFERENT PIECES OF FISHER
        grad = self.get_gradients_sgd(grad, _dμ_dθ, _invC, _invF, d_dμdθ_w, dC_dw)

        updates, state = self._update(grad, state, w)
        w = optax.apply_updates(w, updates) # UPDATE PARAMS

        detF, detC, detinvC, Λ2, r = self._update_history(
            results, (detF, detC, detinvC, Λ2, r), counter, 0)
        if self.validate:
            F, C, invC, *_ = self._get_F_statistics(
                w, key=validation_key, validate=True)
            _Λ2 = self._get_regularisation(C)
            _r = self._get_regularisation_strength(_Λ2, λ, α)
            results = (F, C, invC, _Λ2, _r)
            detF, detC, detinvC, Λ2, r = self._update_history(
                results, (detF, detC, detinvC, Λ2, r), counter, 1)
        _detF = np.linalg.det(results[0])
        patience_counter, counter, _, max_detF, __, best_w = \
            jax.lax.cond(
                np.greater(_detF, max_detF),
                self._update_loop_vars,
                lambda inputs: self._check_loop_vars(inputs, min_iterations),
                (patience_counter, counter, _detF, max_detF, w, best_w))
        return (max_detF, best_w, detF, detC, detinvC, Λ2, r,
                counter + np.int32(1), patience_counter, state, w, rng)

    def _fit_cond(self, inputs, patience, max_iterations):

        return np.logical_and(
            np.less(inputs[-4], patience),
            np.less(inputs[-5], max_iterations))

    def _update_loop_vars(self, inputs):

        patience_counter, counter, detF, max_detF, w, best_w = inputs
        return (np.int32(0), counter, detF, detF, w, w)

    def _check_loop_vars(self, inputs, min_iterations):

        patience_counter, counter, detF, max_detF, w, best_w = inputs
        patience_counter = jax.lax.cond(
            np.greater(counter, min_iterations),
            lambda patience_counter: patience_counter + np.int32(1),
            lambda patience_counter: patience_counter,
            patience_counter)
        return (patience_counter, counter, detF, max_detF, w, best_w)

    def _update_history(self, inputs, history, counter, ind):

        F, C, invC, _Λ2, _r = inputs
        detF, detC, detinvC, Λ2, r = history
        detF = detF.at[counter, ind].set(np.linalg.det(F))
        detC = detC.at[counter, ind].set(np.linalg.det(C))
        detinvC = detinvC.at[counter, ind].set(np.linalg.det(invC))
        Λ2 = Λ2.at[counter, ind].set(_Λ2)
        r = r.at[counter, ind].set(_r)
        return detF, detC, detinvC, Λ2, r

    def _slogdet(self, matrix):
        """Combined summed logarithmic determinant

        Parameters
        ----------
        matrix : float(n, n)
            An n x n matrix to calculate the summed logarithmic determinant of

        Returns
        -------
        float:
            The summed logarithmic determinant multiplied by its sign
        """
        lndet = np.linalg.slogdet(matrix)
        return lndet[0] * lndet[1]

    def _construct_derivatives(self, derivatives):

        return derivatives

    def set_F_statistics(self, w=None, key=None, validate=True):

        if validate and ((not self.validate) and (not self.simulate)):
            validate = False
        if w is not None:
            self.w = w
        self.F, self.C, self.invC, self.dμ_dθ, self.μ, self.F_loss = \
            self._get_F_statistics(key=key, validate=validate)
        self.invF = np.linalg.inv(self.F)

    def _get_F_statistics(self, w=None, key=None, validate=False):

        if w is None:
            w = self.w
        summaries, derivatives = self.get_summaries(
            w=w, key=key, validate=validate)
        return self._calculate_F_statistics(summaries, derivatives)

    @partial(jax.jit, static_argnums=0)
    def _calculate_F_statistics(self, summaries, derivatives):

        # FREEZE gradient calculation for functions
        derivatives = self._construct_derivatives(derivatives)

        μ = np.mean(summaries, axis=0)
        
        # FIXED C
        C = np.cov(summaries, rowvar=False)

        if self.n_summaries == 1:
            C = np.array([[C]])

        invC = np.linalg.inv(C)

        dμ_dθ = np.mean(derivatives, axis=0)

        F = np.einsum("ij,ik,kl->jl", dμ_dθ, invC, dμ_dθ)
        Finv = np.linalg.inv(F)

        return (F, C, invC, dμ_dθ, μ, Finv)
    
    @partial(jax.jit, static_argnums=0)
    def _calculate_mean_dervs(self, summaries, derivatives):

        derivatives = self._construct_derivatives(derivatives)

        # WITH GRADIENT
        dμ_dθ = np.mean(derivatives, axis=0)

        return dμ_dθ
    
    @partial(jax.jit, static_argnums=0)
    def _calculate_Df(self, summaries, derivatives):
        
        # uses a batch of summaries (need to define in next class)
        μ = np.mean(summaries, axis=0)
        return summaries - μ
    
    def _get_Df(self, w, key=None):
        summaries, derivatives = self.get_summaries_sgd(w=w, key=key)
        return self._calculate_Df(summaries, derivatives), (summaries, derivatives)
        
    #@partial(jax.jit, static_argnums=0)
    def _calculate_Cw(self, w, key=None):
        """Estimate derivative of network summary covariance
           w.r.t network weights without using the full fiducial
           dataset
        """
        
        Df_w, (summaries, derivatives) = jax.jacrev(self._get_Df, argnums=0, has_aux=True)(w, key=key)
        
        C_w = jax.tree_map(lambda dfw: 0.5 * np.mean(np.einsum("bi...,bj->bij...", dfw, summaries)
                                        + np.einsum("bi,bj...->bij...", summaries, dfw), axis=0), Df_w)
        
        return C_w
        
        

    @partial(jax.jit, static_argnums=0)
    def _get_regularisation_strength(self, Λ2, λ, α):

        return λ * Λ2 / (Λ2 + np.exp(-α * Λ2))

    @partial(jax.jit, static_argnums=0)
    def _get_regularisation(self, C):

        #if self.no_invC:
        if self.evidence:
            reg = np.trace(C)

        else:
            reg = np.linalg.norm(C - np.eye(self.n_summaries))

        #else:
            #reg = np.linalg.norm(C - np.eye(self.n_summaries)) + \
                #np.linalg.norm(invC - np.eye(self.n_summaries))
        return reg

    def _get_loss(self, w, λ, α, γ, key=None):

        summaries, derivatives = self.get_summaries(w=w, key=key)
        return self._calculate_loss(summaries, derivatives, λ, α, γ)

    def _get_C(self, w, key=None):
        
        # TODO: do we need to do the regularization through the covariance calculation ?
        summaries, derivatives = self.get_summaries_sgd(w=w, key=key)
        
        return np.cov(summaries, rowvar=False)
    

    def _get_loss_dervs(self, w, λ, α, γ, key=None):

        summaries, derivatives = self.get_summaries_sgd(w=w, key=key)
        
        return self._calculate_dμ_dθ_loss(summaries, derivatives, λ, α, γ)

    def _get_loss_reg(self, w, λ, α, γ, key=None):
        
        # no gradients here
        summaries, derivatives = self.get_summaries(w=w, key=key)
        F, C_, invC, dμ_dθ, _, Fw = self._calculate_F_statistics(
            summaries, derivatives)
        
        # gradient is passed through this one
        C = self._get_C(w, key=key)
        
        Λ2 = self._get_regularisation(C) # will ignore invC if correctly set
        
        if self.do_reg:
            r = self._get_regularisation_strength(Λ2, λ, α)
        else:
            r = γ*0.5
        
        return r * Λ2, (F, C, invC, Λ2, r, dμ_dθ)


    # def _calculate_reg_loss(self, summaries, derivatives, λ, α, γ):

    #     F, C, invC, dμ_dθ, _, Fw = self._calculate_F_statistics(
    #         summaries, derivatives)
                
    #     Λ2 = self._get_regularisation(C)
        
    #     if self.do_reg:
    #         r = self._get_regularisation_strength(Λ2, λ, α)
    #     else:
    #         r = γ*0.5
    #     return r * Λ2, (F, C, invC, Λ2, r, dμ_dθ)
    

    def _calculate_dμ_dθ_loss(self, summaries, derivatives, λ, α, γ):
        
        dμ_dθ = self._calculate_mean_dervs(
            summaries, derivatives)

        return dμ_dθ
    
    def get_gradients_sgd(self, reg_grad, dμ_dθ, invC, invF, d_dμdθ_w, dC_dw):
        """ calculate gradient of Fisher wrt network with small 
        batches of simulations
        """
        
        # GET DIFFERENT PIECES OF FISHER
        lastpiece = jax.tree_map(lambda dc: np.einsum("qm,qj,ji...,ik,kl->ml...", dμ_dθ, invC, dc, invC, dμ_dθ), dC_dw)

        grad_Fw = jax.tree_map(lambda d: np.einsum("qm,qj,ji...->mi...", dμ_dθ, invC, d), d_dμdθ_w)
        grad_Fw = jax.tree_map(lambda x,y: 2.*(x-y), grad_Fw, lastpiece)
        grad_Fw = jax.tree_map(lambda x: np.trace(np.einsum("ik,kj...->ij...", invF, x)), grad_Fw)
        grad = jax.tree_map(lambda x,y: x - y, reg_grad, grad_Fw)
        return grad
    

    def get_summaries(self, w=None, key=None, validate=False):

        raise ValueError("`get_summaries` not implemented")
        
    
    def get_summaries_sgd(self, w=None, key=None, validate=False):
        
        raise ValueError("`get_summaries_sgd` not implemented")


    def get_estimate(self, d):
        """Calculate score compressed parameter estimates from network outputs

        Using score compression we can get parameter estimates under the
        transformation

        .. math::
            \\hat{\\boldsymbol{\\theta}}_\\alpha=\\theta^{\\rm{fid}}_\\alpha+
            \\bf{F}^{-1}_{\\alpha\\beta}\\frac{\\partial\\mu_i}{\\partial
            \\theta_\\beta}\\bf{C}^{-1}_{ij}(x(\\bf{w}, \\bf{d})-\\mu)_j

        where :math:`x_j` is the :math:`j` output of the network with network
        parameters :math:`\\bf{w}` and input data :math:`\\bf{d}`.

        Examples
        --------
        Assuming that an IMNN has been fit (as in the example in
        :py:meth:`imnn.imnn._imnn.IMNN.fit`) then we can obtain a
        pseudo-maximum likelihood estimate of some target data (which is
        generated with parameter values μ=1, Σ=2) using

        .. code-block:: python

            rng, target_key = jax.random.split(rng)
            target_data = model_simulator(target_key, np.array([1., 2.]))

            imnn.get_estimate(target_data)
            >>> DeviceArray([0.1108716, 1.7881424], dtype=float32)

        The one standard deviation uncertainty on these parameter estimates
        (assuming the fiducial is at the maximum-likelihood estimate - which we
        know it isn't here) estimated by the square root of the inverse Fisher
        information matrix is

        .. code-block:: python

            np.sqrt(np.diag(imnn.invF))
            >>> DeviceArray([0.31980422, 0.47132865], dtype=float32)

        Note that we can compare the values estimated by the IMNN to the value
        of the mean and the variance of the target data itself, which is what
        the IMNN should be summarising

        .. code-block:: python

            np.mean(target_data)
            >>> DeviceArray(0.10693721, dtype=float32)

            np.var(target_data)
            >>> DeviceArray(1.70872, dtype=float32)

        Note that batches of data can be summarised at once using
        ``get_estimate``. In this example we will draw 10 different values of μ
        from between :math:`-10 < \\mu < 10` and 10 different values of Σ from
        between :math:`0 < \\Sigma < 10` and generate a batch of 10 different
        input data which we can summarise using the IMNN.

        .. code-block:: python

            rng, mean_keys, var_keys = jax.random.split(rng, num=3)

            mean_vals = jax.random.uniform(
                mean_keys, minval=-10, maxval=10, shape=(10,))
            var_vals = jax.random.uniform(
                var_keys, minval=0, maxval=10, shape=(10,))

            np.stack([mean_vals, var_vals], -1)
            >>> DeviceArray([[ 3.8727236,  1.6727388],
                             [-3.1113386,  8.14554  ],
                             [ 9.87299  ,  1.4134324],
                             [ 4.4837523,  1.5812075],
                             [-9.398947 ,  3.5737753],
                             [-2.0789695,  9.978279 ],
                             [-6.2622285,  6.828809 ],
                             [ 4.6470118,  6.0823894],
                             [ 5.7369494,  8.856505 ],
                             [ 4.248898 ,  5.114669 ]], dtype=float32)

            batch_target_keys = np.array(jax.random.split(rng, num=10))

            batch_target_data = jax.vmap(model_simulator)(
                batch_target_keys, (mean_vals, var_vals))

            imnn.get_estimate(batch_target_data)
            >>> DeviceArray([[ 4.6041985,  8.344688 ],
                             [-3.5172062,  7.7219954],
                             [13.229679 , 23.668312 ],
                             [ 5.745726 , 10.020965 ],
                             [-9.734651 , 21.076218 ],
                             [-1.8083427,  6.1901293],
                             [-8.626409 , 18.894459 ],
                             [ 5.7684307,  9.482665 ],
                             [ 6.7861238, 14.128591 ],
                             [ 4.900367 ,  9.472563 ]], dtype=float32)

        Parameters
        ----------
        d : float(None, input_shape)
            Input data to be compressed to score compressed parameter estimates

        Returns
        -------
        float(None, n_params):
            Score compressed parameter estimates

        Methods
        -------
        single_element:
            Returns a single score compressed summary
        multiple_elements:
            Returns a batch of score compressed summaries

        Raises
        ------
        ValueError
            If the Fisher statistics are not set after running ``fit`` or
            ``set_F_statistics``.

        Todo
        ----
        - Do proper checking on input shape (should just be a call to
          ``_check_input``)
        """
        @jax.jit
        def single_element(d):
            """Returns a single score compressed summary

            Parameters
            ----------
            d : float(input_shape)
                Input data to be compressed to score compressed parameter
                estimate

            Returns
            -------
            float(n_params,):
                Score compressed parameter estimate
            """
            return self.θ_fid + np.einsum(
                "ij,kj,kl,l->i",
                self.invF,
                self.dμ_dθ,
                self.invC,
                self.model(self.w, d) - self.μ)

        @jax.jit
        def multiple_elements(d):
            """Returns a batch of score compressed summaries

            Parameters
            ----------
            d : float(None, input_shape)
                Input data to be compressed to score compressed parameter
                estimates

            Returns
            -------
            float(None, n_params):
                Score compressed parameter estimates

            Methods
            -------
            fn:
                Returns the output of the evaluated model
            """
            def fn(d):
                """Returns the output of the evaluated model

                Parameters
                ----------
                d : float(input_shape)
                    Input data to the neural network

                Returns
                -------
                float(None, n_summaries):
                    Neural network output
                """
                return self.model(self.w, d)
            return self.θ_fid + np.einsum(
                "ij,kj,kl,ml->mi",
                self.invF,
                self.dμ_dθ,
                self.invC,
                jax.vmap(fn)(d) - self.μ)

        _check_statistics_set(self.invF, self.dμ_dθ, self.invC, self.μ)
        # check shape: array or graph ?
        if self.dummy_graph_input is None:
          if len(d.shape) == 1:
              return single_element(d)
          else:
              return multiple_elements(d)
        else:
            return single_element(d)

    def _setup_plot(self, ax=None, expected_detF=None, figsize=(5, 15)):
        """Builds axes for history plot

        Parameters
        ----------
        ax : mpl.axes or None, default=None
            An axes object of predefined axes to be labelled
        expected_detF : float or None, default=None
            Value of the expected determinant of the Fisher information to plot
            a horizontal line at to check fitting progress
        figsize : tuple, default=(5, 15)
            The size of the figure to be produced

        Returns
        -------
        mpl.axes:
            An axes object of labelled axes
        """
        if ax is None:
            fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize)
            plt.subplots_adjust(hspace=0.05)
        ax = [x for x in ax] + [ax[2].twinx()]
        if expected_detF is not None:
            ax[0].axhline(expected_detF, linestyle="dashed", color="black")
        ax[0].set_ylabel(r"$|{\bf F}|$")
        ax[1].axhline(1, linestyle="dashed", color="black")
        ax[1].set_ylabel(r"$|{\bf C}|$ and $|{\bf C}^{-1}|$")
        ax[1].set_yscale("log")
        ax[2].set_xlabel("Number of iterations")
        ax[2].set_ylabel(r"$\Lambda_2$")
        ax[3].set_ylabel(r"$r$")
        return ax

    def plot(self, ax=None, expected_detF=None, colour="C0", figsize=(5, 15),
             label="", filename=None, ncol=1):

        if ax is None:
            ax = self._setup_plot(expected_detF=expected_detF, figsize=figsize)
        ax[0].set_xlim(
            0, max(self.history["detF"].shape[0] - 1, ax[0].get_xlim()[-1]))
        ax[0].plot(self.history["detF"], color=colour,
                   label=r"{} $|F|$ (training)".format(label))
        ax[1].set_xlim(
            0, max(self.history["detF"].shape[0] - 1, ax[0].get_xlim()[-1]))
        ax[1].plot(self.history["detC"], color=colour,
                   label=r"{} $|C|$ (training)".format(label))
        ax[1].plot(self.history["detinvC"], linestyle="dotted", color=colour,
                   label=label + r" $|C^{-1}|$ (training)")
        ax[3].set_xlim(
            0, max(self.history["detF"].shape[0] - 1, ax[0].get_xlim()[-1]))
        ax[2].plot(self.history["Λ2"], color=colour,
                   label=r"{} $\Lambda_2$ (training)".format(label))
        ax[3].plot(self.history["r"], color=colour, linestyle="dashed",
                   label=r"{} $r$ (training)".format(label))
        if self.validate:
            ax[0].plot(self.history["val_detF"], color=colour,
                       label=r"{} $|F|$ (validation)".format(label),
                       linestyle="dotted")
            ax[1].plot(self.history["val_detC"], color=colour,
                       label=r"{} $|C|$ (validation)".format(label),
                       linestyle="dotted")
            ax[1].plot(self.history["val_detinvC"],
                       color=colour,
                       label=label + r" $|C^{-1}|$ (validation)",
                       linestyle="dashdot")
            ax[2].plot(self.history["val_Λ2"], color=colour,
                       label=r"{} $\Lambda_2$ (validation)".format(label),
                       linestyle="dotted")
            ax[3].plot(self.history["val_r"], color=colour,
                       label=r"{} $r$ (validation)".format(label),
                       linestyle="dashdot")
        h1, l1 = ax[2].get_legend_handles_labels()
        h2, l2 = ax[3].get_legend_handles_labels()
        ax[0].legend(bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=ncol)
        ax[1].legend(frameon=False, bbox_to_anchor=(1.0, 1.0), ncol=ncol * 2)
        ax[3].legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.05, 1.0),
                     frameon=False, ncol=ncol * 2)

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", transparent=True)
        return ax



class SGDNoiseNumericalGradientIMNN(_SGD_IMNN):
    """Information maximising neural network fit with simulations on-the-fly
    """
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid, δθ,
                 model, optimiser, key_or_state, dummy_graph_input, noise_simulator, 
                 fiducial, derivative,
                 validation_fiducial=None, validation_derivative=None, 
                 n_s_batch=100,
                 n_d_batch=100,
                 no_invC=False, do_reg=True,
                 evidence=False):
        """Constructor method

        Initialises all IMNN attributes, constructs neural network and its
        initial parameter values and creates history dictionary. Also checks
        validity of simulator and sets the ``simulate`` attribute to ``True``.

        Parameters
        ----------
        n_s : int
            Number of simulations used to calculate summary covariance
        n_d : int
            Number of simulations used to calculate mean of summary derivative
        n_params : int
            Number of model parameters
        n_summaries : int
            Number of summaries, i.e. outputs of the network
        input_shape : tuple
            The shape of a single input to the network
        θ_fid : float(n_params,)
            The value of the fiducial parameter values used to generate inputs
        model : tuple, len=2
            Tuple containing functions to initialise neural network
            ``fn(rng: int(2), input_shape: tuple) -> tuple, list`` and the
            neural network as a function of network parameters and inputs
            ``fn(w: list, d: float(None, input_shape)) -> float(None, n_summari
            es)``.
            (Essentibly stax-like, see `jax.experimental.stax <https://jax.read
            thedocs.io/en/stable/jax.experimental.stax.html>`_))
        optimiser : tuple, len=3
            Tuple containing functions to generate the optimiser state
            ``fn(x0: list) -> :obj:state``, to update the state from a list of
            gradients ``fn(i: int, g: list, state: :obj:state) -> :obj:state``
            and to extract network parameters from the state
            ``fn(state: :obj:state) -> list``.
            (See `jax.experimental.optimizers <https://jax.readthedocs.io/en/st
            able/jax.experimental.optimizers.html>`_)
        key_or_state : int(2) or :obj:state
            Either a stateless random number generator or the state object of
            an preinitialised optimiser
        simulator : fn
            A function that generates a single simulation from a random number
            generator and a tuple (or array) of parameter values at which to
            generate the simulations. For the purposes of use in LFI/ABC
            afterwards it is also useful for the simulator to be able to
            broadcast to a batch of simulations on the zeroth axis
            ``fn(int(2,), float([None], n_params)) ->
            float([None], input_shape)``
        dummy_graph_input : jraph.GraphsTuple or 'jax.numpy.DeviceArray'
            Either a (padded) graph input or device array. If supplied ignores 
            `input_shape` parameter
        """
        super().__init__(
            n_s=n_s,
            n_d=n_d,
            n_params=n_params,
            n_summaries=n_summaries,
            input_shape=input_shape,
            θ_fid=θ_fid,
            model=model,
            optimiser=optimiser,
            key_or_state=key_or_state,
            dummy_graph_input=dummy_graph_input,
            no_invC=no_invC,
            do_reg=do_reg,
            evidence=evidence)
        
        self.simulator = noise_simulator #_check_simulator(noise_simulator)
        #self.simulate = True
        self.dummy_graph_input = dummy_graph_input
        self.θ_der = (θ_fid + np.einsum("i,jk->ijk", np.array([-1., 1.]), 
                                        np.diag(δθ) / 2.)).reshape((-1, 2))
        self.δθ = np.expand_dims(
            _check_input(δθ, (self.n_params,), "δθ"), (0, 1))
        
        # NUMERICAL GRADIENT SETUP
        self._set_data(δθ, fiducial, derivative, validation_fiducial,
                       validation_derivative)
        
        # SGD BATCH SIZE
        #self.batch_size = batch_size
        self.n_s_batch = n_s_batch
        self.n_d_batch = n_d_batch


    def _set_data(self, δθ, fiducial, derivative, validation_fiducial,
                  validation_derivative):
        """Checks and sets data attributes with the correct shape
        """
        self.δθ = np.expand_dims(
            _check_input(δθ, (self.n_params,), "δθ"), (0, 1))
        if self.dummy_graph_input is None:
          self.fiducial = fiducial
#             _check_input(
#               fiducial, (self.n_s,) + self.input_shape, "fiducial")
        
          self.derivative = derivative
#             _check_input(
#               derivative, (self.n_d, 2, self.n_params) + self.input_shape,
#               "derivative")
          if ((validation_fiducial is not None)
                  and (validation_derivative is not None)):
              self.validation_fiducial = validation_fiducial
#                 _check_input(
#                   validation_fiducial, (self.n_s,) + self.input_shape,
#                   "validation_fiducial")
              self.validation_derivative = validation_derivative
#                     _check_input(
#                   validation_derivative,
#                   (self.n_d, 2, self.n_params) + self.input_shape,
#                   "validation_derivative")
              self.validate = True
        else:
          self.fiducial = fiducial
          self.derivative = derivative

          if ((validation_fiducial is not None)
                  and (validation_derivative is not None)):
              self.validation_fiducial = validation_fiducial
              self.validation_derivative =  validation_derivative
              self.validate = True


    def _collect_input(self, key, validate=False, 
                       fid_idx=None, derv_idx=None):
        """ Returns validation or fitting sets with noise.
        """
        if validate:
            fiducial = self.validation_fiducial
            derivative = self.validation_derivative
        else:
            fiducial = self.fiducial
            derivative = self.derivative
            
        if fid_idx is not None:
            fiducial = fiducial[fid_idx]
            derivative = np.array(
                np.split(derivative.reshape((self.n_d*2*self.n_params,)+self.input_shape), 
                                      self.n_d))[derv_idx[:self.n_d_batch]].reshape((self.n_d_batch*2*self.n_params,) + self.input_shape)
            _n_s = self.n_s_batch
            _n_d = self.n_d_batch
        
        else:
            _n_s = self.n_s
            _n_d = self.n_d
            
        # add noise to data and make cuts
        keys = np.array(jax.random.split(key, num=_n_s))
        fiducial = jax.vmap(self.simulator)(keys, fiducial)
        
        _shape = derivative.shape
        derivative = jax.vmap(self.simulator)(
                np.repeat(keys[:_n_d], 2*self.n_params, axis=0),
                derivative.reshape(
                      (_n_d * 2 * self.n_params,) + self.input_shape)).reshape(_shape)
                      
        return fiducial, derivative

    def _get_fitting_keys(self, rng):
        """Generates random numbers for simulation

        Parameters
        ----------
        rng : int(2,)
            A random number generator

        Returns
        -------
        int(2,), int(2,), int(2,)
            A new random number generator and random number generators for
            fitting (and validation)
        """
        return jax.random.split(rng, num=3)

    def get_summaries(self, w, key=None, validate=False):
        """Gets all network outputs and derivatives wrt model parameters
        """
        d, d_mp = self._collect_input(key, validate=validate)
        
        
        if self.dummy_graph_input is None:
          _model = lambda d: self.model(w, d)
          x = jax.vmap(_model)(d)
          x_mp = np.reshape(
              jax.vmap(_model)(
                    d_mp.reshape(
                      (self.n_d * 2 * self.n_params,) + self.input_shape)),
              (self.n_d, 2, self.n_params, self.n_summaries))

        else:
          # if operating on graph data, we need to vmap the implicit
          # batch dimension
          _model = lambda d: self.model(w, d)
          x = jax.vmap(_model)(d)
          x_mp = np.reshape(
              jax.vmap(_model)(d_mp),
              (self.n_d, 2, self.n_params, self.n_summaries))

        return x, x_mp
    

    
        
    def get_summaries_sgd(self, w=None, key=None, validate=False):
        """Gets small batch of network outputs and derivatives wrt model parameters
        """
        #d, d_mp = self._collect_input(key, validate=validate)
        
        # get only a batch_size set of the fiducial data
        #idx = jax.random.choice(key, np.arange(self.n_s), shape=(self.n_s_batch,), replace=False)
        #d = d[idx]
        # now do derivatives
        #idx = jax.random.choice(key, np.arange(self.n_d), shape=(self.n_d_batch,), replace=False)
        
        #d_mp = np.array(
        #        np.split(d_mp.reshape((self.n_d*2*self.n_params,)+self.input_shape), 
        #                              self.n_d))[idx[:self.n_d_batch]].reshape((self.n_d_batch*2*self.n_params,) + self.input_shape)
        
        fid_idx = jax.random.choice(key, np.arange(self.n_s), shape=(self.n_s_batch,), replace=False)
        derv_idx = jax.random.choice(key, np.arange(self.n_d), shape=(self.n_d_batch,), replace=False)

        d, d_mp = self._collect_input(key, validate=validate, fid_idx=fid_idx, derv_idx=derv_idx)
        
        if self.dummy_graph_input is None:
          _model = lambda d: self.model(w, d)
          x = jax.vmap(_model)(d)
            
          x_mp = np.reshape(
              jax.vmap(_model)(
                    d_mp.reshape(
                      (self.n_d_batch * 2 * self.n_params,) + self.input_shape)),
              (self.n_d_batch, 2, self.n_params, self.n_summaries))
            

        else:
          # if operating on graph data, we need to vmap the implicit
          # batch dimension
          _model = lambda d: self.model(w, d)
          x = jax.vmap(_model)(d)
          x_mp = np.reshape(
              jax.vmap(_model)(d_mp),
              (self.n_d, 2, self.n_params, self.n_summaries))

        return x, x_mp
        

    def _construct_derivatives(self, x_mp):
        """Builds derivatives of the network outputs wrt model parameters
        """
        return np.swapaxes(x_mp[:, 1] - x_mp[:, 0], 1, 2) / self.δθ
    


class BatchedNoiseNumericalGradientIMNN(_SGD_IMNN):
    """Information maximising neural network fit with simulations on-the-fly
    """
    def __init__(self, n_s, n_d, n_params, n_summaries, input_shape, θ_fid, δθ,
                 model, optimiser, key_or_state, dummy_graph_input, noise_simulator, 
                 fiducial, derivative,
                 validation_fiducial=None, validation_derivative=None, 
                 n_per_device=100,
                 n_s_batch=100,
                 n_d_batch=100,
                 no_invC=False, do_reg=True,
                 evidence=False):
        """Constructor method

        Initialises all IMNN attributes, constructs neural network and its
        initial parameter values and creates history dictionary. Also checks
        validity of simulator and sets the ``simulate`` attribute to ``True``.

        Parameters
        ----------
        n_s : int
            Number of simulations used to calculate summary covariance
        n_d : int
            Number of simulations used to calculate mean of summary derivative
        n_params : int
            Number of model parameters
        n_summaries : int
            Number of summaries, i.e. outputs of the network
        input_shape : tuple
            The shape of a single input to the network
        θ_fid : float(n_params,)
            The value of the fiducial parameter values used to generate inputs
        model : tuple, len=2
            Tuple containing functions to initialise neural network
            ``fn(rng: int(2), input_shape: tuple) -> tuple, list`` and the
            neural network as a function of network parameters and inputs
            ``fn(w: list, d: float(None, input_shape)) -> float(None, n_summari
            es)``.
            (Essentibly stax-like, see `jax.experimental.stax <https://jax.read
            thedocs.io/en/stable/jax.experimental.stax.html>`_))
        optimiser : tuple, len=3
            Tuple containing functions to generate the optimiser state
            ``fn(x0: list) -> :obj:state``, to update the state from a list of
            gradients ``fn(i: int, g: list, state: :obj:state) -> :obj:state``
            and to extract network parameters from the state
            ``fn(state: :obj:state) -> list``.
            (See `jax.experimental.optimizers <https://jax.readthedocs.io/en/st
            able/jax.experimental.optimizers.html>`_)
        key_or_state : int(2) or :obj:state
            Either a stateless random number generator or the state object of
            an preinitialised optimiser
        simulator : fn
            A function that generates a single simulation from a random number
            generator and a tuple (or array) of parameter values at which to
            generate the simulations. For the purposes of use in LFI/ABC
            afterwards it is also useful for the simulator to be able to
            broadcast to a batch of simulations on the zeroth axis
            ``fn(int(2,), float([None], n_params)) ->
            float([None], input_shape)``
        dummy_graph_input : jraph.GraphsTuple or 'jax.numpy.DeviceArray'
            Either a (padded) graph input or device array. If supplied ignores 
            `input_shape` parameter
        """
        super().__init__(
            n_s=n_s,
            n_d=n_d,
            n_params=n_params,
            n_summaries=n_summaries,
            input_shape=input_shape,
            θ_fid=θ_fid,
            model=model,
            optimiser=optimiser,
            key_or_state=key_or_state,
            dummy_graph_input=dummy_graph_input,
            no_invC=no_invC,
            do_reg=do_reg,
            evidence=evidence)
        
        self.simulator = noise_simulator #_check_simulator(noise_simulator)
        #self.simulate = True
        self.dummy_graph_input = dummy_graph_input
        self.θ_der = (θ_fid + np.einsum("i,jk->ijk", np.array([-1., 1.]), 
                                        np.diag(δθ) / 2.)).reshape((-1, 2))
        self.δθ = np.expand_dims(
            _check_input(δθ, (self.n_params,), "δθ"), (0, 1))
        
        self.n_per_device = n_per_device
        self._set_shapes() # get batch info
        
        # NUMERICAL GRADIENT SETUP
        self._set_data(δθ, fiducial, derivative, validation_fiducial,
                       validation_derivative)
        
        # SGD BATCH SIZE
        #self.batch_size = batch_size
        self.n_s_batch = n_s_batch
        self.n_d_batch = n_d_batch
    
    def _set_shapes(self):
        self.fid_splits = self.n_s // self.n_per_device
        self.derv_splits = self.n_d // self.n_per_device


    def _set_data(self, δθ, fiducial, derivative, validation_fiducial,
                  validation_derivative):
        """Checks and sets data attributes with the correct shape
        """
        self.δθ = np.expand_dims(
            _check_input(δθ, (self.n_params,), "δθ"), (0, 1))
        if self.dummy_graph_input is None:
          self.fiducial = fiducial
          self.derivative = derivative
            

          if ((validation_fiducial is not None)
                  and (validation_derivative is not None)):
              self.validation_fiducial = validation_fiducial
              self.validation_derivative = validation_derivative
              self.validate = True
        else:
          self.fiducial = fiducial
          self.derivative = derivative

          if ((validation_fiducial is not None)
                  and (validation_derivative is not None)):
              self.validation_fiducial = validation_fiducial
              self.validation_derivative =  validation_derivative
              self.validate = True
        


    def _collect_input(self, key, validate=False, 
                       fid_idx=None, derv_idx=None):
        """ Returns validation or fitting sets with noise.
        """
        if validate:
            fiducial = self.validation_fiducial
            derivative = self.validation_derivative
        else:
            fiducial = self.fiducial
            derivative = self.derivative
            
        if fid_idx is not None:
            fiducial = fiducial[fid_idx]
            derivative = np.array(
                np.split(derivative.reshape((self.n_d*2*self.n_params,)+self.input_shape), 
                                      self.n_d))[derv_idx[:self.n_d_batch]].reshape((self.n_d_batch*2*self.n_params,) + self.input_shape)
            _n_s = self.n_s_batch
            _n_d = self.n_d_batch
        
        else:
            _n_s = self.n_s
            _n_d = self.n_d
            
        # add noise to data and make cuts
        keys = np.array(jax.random.split(key, num=_n_s))
        fiducial = jax.vmap(self.simulator)(keys, fiducial)
        
        _shape = derivative.shape
        derivative = jax.vmap(self.simulator)(
                np.repeat(keys[:_n_d], 2*self.n_params, axis=0),
                derivative.reshape(
                      (_n_d * 2 * self.n_params,) + self.input_shape)).reshape(_shape)
                      
        return fiducial, derivative

    def _get_fitting_keys(self, rng):
        """Generates random numbers for simulation

        Parameters
        ----------
        rng : int(2,)
            A random number generator

        Returns
        -------
        int(2,), int(2,), int(2,)
            A new random number generator and random number generators for
            fitting (and validation)
        """
        return jax.random.split(rng, num=3)

    def get_summaries(self, w, key=None, validate=False):
        """Gets all network outputs and derivatives wrt model parameters
        """
        d, d_mp = self._collect_input(key, validate=validate)
        
        d = np.split(d, self.fid_splits) # split into batches
        d_mp = np.split(d_mp.reshape((self.n_d * 2 * self.n_params,) + self.input_shape), self.derv_splits)
        
        
        if self.dummy_graph_input is None:
          _model = lambda d: self.model(w, d)
          x = np.concatenate([jax.vmap(_model)(_d) for _d in d])
          x_mp = np.concatenate([jax.vmap(_model)(_dmp) for _dmp in d_mp])
          x_mp = np.reshape(x_mp, (self.n_d, 2, self.n_params, self.n_summaries))
        

        else:
          # if operating on graph data, we need to vmap the implicit
          # batch dimension
          _model = lambda d: self.model(w, d)
          x = jax.vmap(_model)(d)
          x_mp = np.reshape(
              jax.vmap(_model)(d_mp),
              (self.n_d, 2, self.n_params, self.n_summaries))

        return x, x_mp
    

    
        
    def get_summaries_sgd(self, w=None, key=None, validate=False):
        """Gets small batch of network outputs and derivatives wrt model parameters
        """
        
        fid_idx = jax.random.choice(key, np.arange(self.n_s), shape=(self.n_s_batch,), replace=False)
        derv_idx = jax.random.choice(key, np.arange(self.n_d), shape=(self.n_d_batch,), replace=False)

        d, d_mp = self._collect_input (key, validate=validate, fid_idx=fid_idx, derv_idx=derv_idx)
        
        if self.dummy_graph_input is None:
          _model = lambda d: self.model(w, d)
          x = jax.vmap(_model)(d)
            
          x_mp = np.reshape(
              jax.vmap(_model)(
                    d_mp.reshape(
                      (self.n_d_batch * 2 * self.n_params,) + self.input_shape)),
              (self.n_d_batch, 2, self.n_params, self.n_summaries))
            

        else:
          # if operating on graph data, we need to vmap the implicit
          # batch dimension
          _model = lambda d: self.model(w, d)
          x = jax.vmap(_model)(d)
          x_mp = np.reshape(
              jax.vmap(_model)(d_mp),
              (self.n_d, 2, self.n_params, self.n_summaries))

        return x, x_mp
        

    def _construct_derivatives(self, x_mp):
        """Builds derivatives of the network outputs wrt model parameters
        """
        return np.swapaxes(x_mp[:, 1] - x_mp[:, 0], 1, 2) / self.δθ