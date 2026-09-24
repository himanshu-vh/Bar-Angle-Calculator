import numpy as np
try:
    import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover
    plt = None
import emcee
import corner

class BarAngleFitter:
    """
    MCMC fitter for bar angle, parallax zero-point, and optionally bar model parameters.
    
    Parameters:
    -----------
    parallax_data : array-like
        Measured parallax values (mas)
    parallax_error : array-like
        Measurement errors on parallax (mas)
    l_values : array-like
        Galactic longitude values (degrees)
    b_values : array-like or float
        Galactic latitude values (degrees). Can be single value or array.
    model_params : dict
        Bar model parameters (sigma_x, sigma_y, sigma_z, r_E, s_max). For fixed params, provide values. 
        For fitted params, provide dict with 'mean' and 'std' keys as a prior. 
        Example: {'sigma_x': {'mean': 0.67, 'std': 0.1}, 'sigma_y': 0.29, ...}
    fit_params : list, optional
        List of parameter names to fit. Default: ['bar_angle', 'zp']
        Can include: 'bar_angle', 'zp', 'sigma_x', 'sigma_y', 'sigma_z', 'r_E'
    """
    
    def __init__(self, parallax_data, parallax_error, l_values, b_values, model_params, fit_params=None):
        self.parallax_data = np.array(parallax_data)
        self.parallax_error = np.array(parallax_error)
        self.l_values = np.array(l_values)
        
        # Handle single b value or array
        if np.isscalar(b_values):
            self.b_values = np.full_like(l_values, b_values, dtype=float)
        else:
            self.b_values = np.array(b_values)
            
        self.model_params = model_params
        self.sampler = None
        self.samples = None
        
        # Define which parameters to fit
        if fit_params is None:
            self.fit_params = ['bar_angle', 'zp']
        else:
            self.fit_params = fit_params    
        
        self.all_param_names = ['bar_angle', 'zp', 'sigma_x', 'sigma_y', 'sigma_z', 'r_E']

        # Import the model function
        from bar_parallax_analytic_model import bar_parallax3D
        self.bar_parallax3D = bar_parallax3D

    def _theta_to_dict(self, theta):
        """Convert theta array to parameter dictionary"""
        params = {}
        for i, param_name in enumerate(self.fit_params):
            params[param_name] = theta[i]
        return params
    def _get_model_params(self, theta):
        """Get full model parameters, combining fitted and fixed values"""
        fitted = self._theta_to_dict(theta)
        
        model_kwargs = {}
        for param in ['sigma_x', 'sigma_y', 'sigma_z', 'r_E', 's_max']:
            if param in fitted:
                model_kwargs[param] = fitted[param]
            else:
                # Get fixed value
                val = self.model_params.get(param)
                if isinstance(val, dict):
                    model_kwargs[param] = val['mean']  # Use mean if prior given
                else:
                    model_kwargs[param] = val
                    
        return model_kwargs

    def model_parallax(self, theta):
        """Compute model parallax for given parameters"""
        fitted = self._theta_to_dict(theta)
        bar_angle = fitted.get('bar_angle', 25.0)
        zp = fitted.get('zp', 0.0)
        
        model_kwargs = self._get_model_params(theta)
        
        model = []
        for l, b in zip(self.l_values, self.b_values):
            plx = self.bar_parallax3D(l, b, bar_angle, **model_kwargs)
            model.append(plx + zp)
        return np.array(model)
    
    def log_likelihood(self, theta):
        model = self.model_parallax(theta)
        if not np.all(np.isfinite(model)):
            return -np.inf          # reject unphysical parameter combos        
        # Chi-squared
        chi2 = np.sum(((self.parallax_data - model) / self.parallax_error) ** 2)
        return -0.5 * chi2
    
    def log_prior(self, theta):
        """Define priors on parameters (Gaussian or uniform)"""
        log_prob = 0.0
        
        for i, param_name in enumerate(self.fit_params):
            value = theta[i]
            
            # Get prior specification
            if param_name == 'bar_angle':
                # Use uniform or Gaussian prior
                prior_spec = self.priors_range.get(param_name, (0, 90)) if self.priors_range else (0, 90)
            elif param_name == 'zp':
                prior_spec = self.priors_range.get(param_name, (-0.1, 0.1)) if self.priors_range else (-0.1, 0.1)
            else:
                # For model parameters, get from model_params
                #prior_spec = self.model_params.get(param_name)
                if self.priors_range and param_name in self.priors_range:
                    prior_spec = self.priors_range[param_name]
                else:
                    prior_spec = self.model_params.get(param_name)
                
            # Evaluate prior
            if isinstance(prior_spec, dict) and 'mean' in prior_spec and 'std' in prior_spec:
                # Gaussian prior
                mean = prior_spec['mean']
                std = prior_spec['std']
                log_prob += -0.5 * ((value - mean) / std) ** 2
            elif isinstance(prior_spec, (tuple, list)) and len(prior_spec) == 2:
                # Uniform prior
                if not (prior_spec[0] <= value <= prior_spec[1]):
                    return -np.inf
            else:
                # No prior specified, use broad uniform
                pass
                
        return log_prob
    
    def log_probability(self, theta):
        """Calculate log-posterior (prior + likelihood)"""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
        
    def run_mcmc(self, n_walkers=5, n_steps=5000, n_burn=1000, thin=1, initial_guess=None, priors_range=None):
        self.priors_range = priors_range or {}
        self.n_burn = n_burn
        self.thin = thin
        n_dim = len(self.fit_params)

        # Convert initial_guess to array
        if isinstance(initial_guess, dict):
            initial_array = [initial_guess[p] for p in self.fit_params]
        elif initial_guess is None:
            # Use default values
            initial_array = []
            for param in self.fit_params:
                if param == 'bar_angle':
                    initial_array.append(10.0)
                elif param == 'zp':
                    initial_array.append(0.0)
                else:
                    val = self.model_params.get(param)
                    if isinstance(val, dict):
                        initial_array.append(val['mean'])
                    else:
                        initial_array.append(val)
        else:
            initial_array = initial_guess
            
        initial_array = np.array(initial_array)

        # Initialize walkers with proper perturbations
        # Use parameter-specific scales for perturbations
        perturbation_scales = []
        for param in self.fit_params:
            if param == 'bar_angle':
                perturbation_scales.append(0.1)  # ~0.1 degrees
            elif param == 'zp':
                perturbation_scales.append(0.001)  # ~0.001 mas
            elif param in ['sigma_x', 'sigma_y', 'sigma_z']:
                perturbation_scales.append(0.01)  # ~0.01 kpc
            elif param == 'r_E':
                perturbation_scales.append(0.01)  # ~0.01 kpc
            else:
                # Default: 1% of value or 0.01 if value is zero
                val = initial_array[len(perturbation_scales)]
                perturbation_scales.append(max(0.01 * abs(val), 0.01))
        
        perturbation_scales = np.array(perturbation_scales)
        
        pos = initial_array + perturbation_scales * np.random.randn(n_walkers, n_dim)
        
        self.sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.log_probability)
        self.sampler.run_mcmc(pos, n_steps, progress=True)
        
        self.samples = self.sampler.get_chain(discard=n_burn, thin=thin, flat=True)
        self.chains  = self.sampler.get_chain()
        #tau = self.sampler.get_autocorr_time()
        #print(tau)
        return self.samples
        
    def get_results(self):
        if self.samples is None:
            raise ValueError("Must run MCMC first!")
        
        results = {}        

        for i, param_name in enumerate(self.fit_params):
            samples_i = self.samples[:, i]
            results[param_name] = {
                'median': np.median(samples_i),
                'mean': np.mean(samples_i),
                'std': np.std(samples_i),
                'p16': np.percentile(samples_i, 16),
                'p84': np.percentile(samples_i, 84)
            }
        
        return results
    
    def plot_chains(self, filename=None):
        if self.sampler is None:
            raise ValueError("Must run MCMC first!")
        
        chain = self.sampler.get_chain()
        n_params = len(self.fit_params)
        
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params), sharex=True)
        
        for i in range(n_params):
            ax = axes[i]
            ax.plot(chain[:, :, i], "k", alpha=0.3)
            ax.axvline(self.n_burn, color='red', linestyle='--', 
                      label='Burn-in' if i == 0 else '')
            ax.set_ylabel(self.fit_params[i])
            if i == 0:
                ax.legend()
        
        axes[-1].set_xlabel("Step Number")
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300)
        plt.show()
    
    def plot_corner(self, truths=None, filename=None):
        if self.samples is None:
            raise ValueError("Must run MCMC first!")
        # if truths is None, set to median values
        if truths is None:
            truths = [np.median(self.samples[:, i]) for i in range(len(self.fit_params))]
                
        fig = corner.corner(
            self.samples,
            labels=self.fit_params,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            truths=truths
        )
        
        if filename:
            plt.savefig(filename, dpi=300)
        plt.show()
    
    def plot_fit(self, filename=None):
        if self.samples is None:
            raise ValueError("Must run MCMC first!")
        
        # Get best-fit parameters
        best_fit_theta = np.median(self.samples, axis=0)
        fitted = self._theta_to_dict(best_fit_theta)
        
        # Compute best-fit model
        model_fit = self.model_parallax(best_fit_theta)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(self.l_values, self.parallax_data, 
                   yerr=self.parallax_error,
                   fmt='o-', color='green', alpha=0.8, label='Data', 
                   capsize=5, markersize=6)
        
        ax.plot(self.l_values, model_fit, 
               's--', color='blue', alpha=0.7, markersize=6,
               label=f'Best fit: α={fitted["bar_angle"]:.2f}°, zp={fitted["zp"]:.4f} mas')
        
        ax.set_xlabel('Galactic Longitude (degrees)')
        ax.set_ylabel('Parallax (mas)')
        ax.set_title('Data vs Best-Fit Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(1/8.2, color='gray', linestyle='--', alpha=0.5)

        axsec = ax.secondary_yaxis('right', functions=(lambda x: 1/x, lambda x: 1/x))
        axsec.set_ylabel('Distance (kpc)')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300)
        plt.show()
    
    def plot_results(self, truths=None):
        """Plot all diagnostic plots."""
        self.plot_chains()
        self.plot_corner(truths=truths)
        self.plot_fit()