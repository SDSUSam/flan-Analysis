import numpy as np
from scipy.stats import linregress
import sympy as sp
import matplotlib.pyplot as plt
import flan_plots
from scipy.optimize import curve_fit
    



class DiffusionPlot:

    # We go ahead and define the global variable of the radial grid here. 

    
    def __init__(self, flan_plotter):
        self.fp = flan_plots.FlanPlots(flan_plotter) 
        self.x, _, _ = self.fp.load_cell_centers()
        self._spline_cache = {}  # initialize cache

        
    
    def poloid_av_vel(self, radial_index, frame_index):
        """
        Helper function used to calculate average non-zero velocity at a given radial index.
        
        Inputs: 
            - radial_index (int) : Index of the radial coordinate in the simulation.   
            - frame_index (int)  : Index of the frame of the simulation. 
    
        Outputs:
            - average velocity (double) : Average of the non-zero velocity along the y direction

        """
        
        vel_data = self.fp.load_data_frame("imp_vX", frame=frame_index)
    
        total = 0.0
        count = 0.0
    
        for i in range(vel_data.shape[1]):
            if vel_data[radial_index][i][8] != 0:
                total += vel_data[radial_index][i][8]
                count += 1
    
        return total / count if count > 0 else 0.0
    
    def rAvg_density(self, radial_index, arr):
        """
        Helper function to compute average non-zero density in
        the poloidal direction at a given radial index. If the 
        poloidal axis span has 0 density, returns 0.  
        
        
        Inputs: 
            - radial_index (int)   : Position in the x-axis
            - arr          (array) : Array of density/coords. Should be 3D. 
                                     To be more specific, this takes output 
                                     from the fp.load_data_frame function

        Outputs: 
            - average density (double) : density averaged over the y-axis
        """
        
        total_radial_density = 0.0
        count = 0
    
        for i in range(arr.shape[1]):
            if arr[radial_index][i][8] != 0: # We exclude the non-zero densities
                total_radial_density += arr[radial_index][i][8]
                count += 1
    
        return total_radial_density / count if count > 0 else 0.0 



    
    def spline_coefficients(self, frame_index):
        """
        Natural cubic spline interpolation for smoothing density data.
        We want to make splines between the values of the average radial 
        density - thus we need to take in the average radial density 
        as data points (as a function of the radial_index). 
        
        Inputs: 
            - frame_index  (int)   : Frame number of the simulation
        
        Outputs:
            - S (ndarray): Coefficients of the cubic splines in [a, b, c, d] form. 
                Note that this is an n x 4 array, where n is the number of radial cells
        """
        
        
        n = len(self.x) - 1  
        
        density_array3d = self.fp.load_data_frame("imp_density", frame=frame_index)
        
        a = np.zeros(n+1)
        
        
        # Set the array 'a' as the function values (coefficients of zeroth-degree term)
        for i in range(len(self.x)): 
            a[i]=self.rAvg_density(i,density_array3d)
        
        # Because the step sizes are unfortunately not constant, it doesn't hurt to define them.
        h = np.array([self.x[i+1] - self.x[i] for i in range(n)])

        
        # Compute alpha, the convenient substitute for the expression spit out by the 
        # linear system that lets you solve for an individual coefficient. 
        
        alpha = np.zeros(n)
        for i in range(1, n):
            alpha[i] = (3/h[i]) * (a[i+1] - a[i]) - (3/h[i-1]) * (a[i] - a[i-1])
        
        # Initialize arrays
        l = np.zeros(n+1)
        u = np.zeros(n+1)
        z = np.zeros(n+1)
        c = np.zeros(n+1)
        b = np.zeros(n)
        d = np.zeros(n)
        
        # The following is an intricate implementation of the Thomas algorithm, 
        # which supposedly is O(n) and not O(n^3). 
        # The code still takes a while though.
        
        l[0] = 1
        u[0] = 0
        z[0] = 0
        
        for i in range(1, n):
            l[i] = 2 * (self.x[i+1] - self.x[i-1]) - h[i-1] * u[i-1]
            u[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
        

        l[n] = 1
        z[n] = 0
        c[n] = 0
        
        # Back substitution
        for j in range(n):
            i = n - 1 - j  # Equivalent to reversed index: j = 0 → i = n-1 
            c[i] = z[i] - u[i] * c[i + 1]
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        
            
        # Construct the spline coefficient array: [a_i, b_i, c_i, d_i]
        S = np.array([[a[i], b[i], c[i], d[i]] for i in range(n)])
        
        return S

    def coefficients(self, frame_index):
        """
        Helper function to avoid calling the thomas algorithm more than is necessary. 
        This function stores the values of the coefficients for a single frame. 

        Inputs: 
            - frame_index 

        Outputs: 
            - n x 4 array of the spline polynomial coefficients.
        """
        if frame_index not in self._spline_cache:
            self._spline_cache[frame_index] = self.spline_coefficients(frame_index)
    
        return self._spline_cache[frame_index]
        
        
    def spline_function(self, radial_index, frame_index):
        """
        Constructs a single cubic polynomial function for the spline interval specified
    
        Inputs:
            - radial_index (int): Index of the spline interval (must be < len(x)-1)
            - frame_index  (int): Frame of the simulation to pull data from
    
        Ouputs:
            - A polynomial function associated within [radial_index, radial_index+dx] 
        """
        
        coef = self.coefficients(frame_index=frame_index)
    
        # Instead of calling the spline coefficients function over and over, we define them here. 
        a = coef[radial_index][0]
        b = coef[radial_index][1]
        c = coef[radial_index][2]
        d = coef[radial_index][3]
        
    
        # Define and return the cubic polynomial S_i(x)
        def polynomial(t, a=a, b=b, c=c, d=d, ri=radial_index):
            xc = t - self.x[ri]
            return a + b * xc + c * xc**2 + d * xc**3

    
        return polynomial

  
  
    def piecewise_spline(self, frame_index): 
        """
        Complete piecewise spline function for average particle density. 
        
        Inputs:    
            - radial_index:
            - frame_index: 
        
        Outputs: 
            - Array of spline piecewise components, together forming 
                the piecewise spline function. To be used when calculating
                the derivative of density. Should be much smoother. 
        """
    
        coeffs = self.coefficients(frame_index).shape[0]
        spline_funcs = []
        
        for i in range(coeffs):
           spline_funcs.append(self.spline_function(i, frame_index))
        
        return spline_funcs



    def spline_chooser(self, radial_index, frame_index):
        """
        Helper function to choose a spline function depending 
        on where in the radial axis a function is needed

        Inputs:
            - radial_index
            - frame_index

        Outputs: 
            - a specific spline function output

        """
        spline_list = self.piecewise_spline(frame_index) 

        returned_spline = spline_list[radial_index]

        return returned_spline
    
    
    def den_deriv(self, radial_index, frame_index, spline=True):
        """
        Helper function used to compute radial derivative of impurity density.
        This function takes the difference between two density values and then divides
        it by the difference in cell positions. 

        -----------------------------------------------------------------
        This may not work as I intend it to. The density pulled is 
        non-dimensionalized right? Need to add dimensionality component before 
        dividing the density difference by radial difference. 
        -----------------------------------------------------------------

        Inputs: 
            - radial_index (int) : Index of the radial coordinate in the simulation.   
            - frame_index (int)  : Index of the frame of the simulation.

        Outputs: 
            - Density derivative (double) : Calculated by taking the difference  
            -    between the density at the given radial index and the density 
            -    at the next radial index, then dividing by 0.0005746. This is the 
            -    difference between successive radial grid lines. Note that at the 
            -    right edge of the simulation, the derivative will be 0 by convention
        
        """
        
        density_array = self.fp.load_data_frame("imp_density", frame=frame_index)
        t = sp.Symbol('t')

        # Calculate the derivatives of average densities via numerical analysis
        if spline:
            """
                Because splines are symbolically way simpler to take derivatives of,
                we make use of sympy to do so. Then, we can just take the specific 
                radial element plug it in to find the derivative. 
                
            """
            # Get spline coefficients at the interval
            if radial_index < len(self.x)-1:
                a, b, c, d = self.coefficients(frame_index)[radial_index]
            else: 
                a, b, c, d = [0,0,0,0]
                
            x0 = self.x[radial_index]
            dx = t - x0
    
            spline_expr = a + b*dx + c*dx**2 + d*dx**3
    
            # Differentiate via sympy. I'm sure there's a better way to do this...
            spline_deriv = sp.diff(spline_expr, t)
    
            # Evaluate at t = x0 (which is where dx = 0)
            return float(spline_deriv.subs(t, x0))

        else:
            # Sorry in advance for how awful this looks
            if radial_index == 0:
                deriv = (-3*self.rAvg_density(radial_index, density_array)+4*self.rAvg_density(radial_index+1, density_array)-self.rAvg_density(radial_index+2, density_array))/(2*(self.x[radial_index+1]-self.x[radial_index]))
            elif 0 < radial_index < len(self.x) - 1:
                deriv = (self.rAvg_density(radial_index+1, density_array)-self.rAvg_density(radial_index-1, density_array))/(2*(self.x[radial_index+1]-self.x[radial_index]))
            else:
                deriv = (3*self.rAvg_density(radial_index, density_array)-4*self.rAvg_density(radial_index-1, density_array)+self.rAvg_density(radial_index-2, density_array))/(2*(self.x[radial_index]-self.x[radial_index-1]))
            return deriv
        
        
    
    def frame_radial_diffusion(self, radial_index, frame_index, spline=True):
        """
        Calculate diffusion coefficient at a specific radial index. 

        Inputs: 
            - radial_index (int) : Index of the radial coordinate in the simulation.   
            - frame_index (int)  : Index of the frame of the simulation.
        
        Outputs: 
            - Returns the diffusion coefficient from Fick's law. 

            ---------------------------------
            does frame_index=frame_index work?
            ---------------------------------
            
        """
        
        dens_array = self.fp.load_data_frame("imp_density", frame=frame_index)
        
        avg_velocity = self.poloid_av_vel(radial_index=radial_index, frame_index=frame_index)

        # Choose if you want to use spline averaged data or raw averages. 
        if spline==True: 
            spline_list = self.piecewise_spline(frame_index)
            if radial_index < len(self.x)-1:
                avg_density = spline_list[radial_index](self.x[radial_index])
            else: 
                avg_density = 0
        else:
            avg_density = self.rAvg_density(radial_index, dens_array)


        # Choose if you want to use the splined derivative. 
        if spline==True: 
            density_grad = self.den_deriv(radial_index, frame_index, spline=True)
        else: 
            density_grad = self.den_deriv(radial_index, frame_index, spline=False)

        
        if density_grad != 0:
            return -avg_velocity * avg_density / density_grad
        else:
            return 0.0
    
    
    def frame_diffusion_array(self, frame_index, spline=True):
        """
        Return array of diffusion coefficients for each radial index.

        Inputs:
            - frame_index (int)  : Index of the frame of the simulation.
        
        Outputs: 
            - diffusion_array (array) : Array containing diffusion coefficients
            -                           for each radial index. 
        
        """
        density_array = self.fp.load_data_frame("imp_density", frame=frame_index)
        diffusion_array = np.zeros(density_array.shape[0])
    
        for i in range(density_array.shape[0]):
            diffusion_array[i] = self.frame_radial_diffusion(radial_index=i, frame_index=frame_index, spline=spline)
    
        return diffusion_array
    
    
    def radial_diffusion_plot(self, frame_index, species, spline):
        """
        Plot diffusion coefficient vs radial coordinate.

        Inputs: 
            - frame_index (int)  : Index of the frame of the simulation.
            - species (str)      : String containing atomic symbol of impurity
                                   as well as the charge state. 

        Outputs:
            - Plot : Plots the diffusion coefficient vs. radial distance 
                     from the center stack.
        
        """
        
        y = self.frame_diffusion_array(frame_index, spline = spline)
    
        plt.plot(self.x, y)
        plt.xlabel("Radial length (m)")
        plt.ylabel("Diffusion Coefficient (m^2 * s^-1)")
        plt.title(f"Radial Diffusion for {species} at {frame_index*0.5} us after injection" )# For timeframe - x (Input fix here)
        plt.grid(True)
        plt.show()

    
    def diffusion_animation(self, species, num_frames=200, interval=100, save_path=None, spline=True):
        """
        Animate the radial diffusion over time.
        
        Inputs:
            - species (str): Species label (e.g., "Tungsten 3+")
            - num_frames (int): Number of simulation frames to animate
            - interval (int): Delay between frames in milliseconds
            - save_path (str or None): If provided, path to save the animation (e.g., "diffusion.mp4" or "diffusion.gif")
        """
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots()
        
        line, = ax.plot([], [], lw=2)

        ax.set_xlim(np.min(self.x), np.max(self.x))
        ax.set_ylim(-2500, 2500)  # You might want to tune this based on your diffusion range
        ax.set_xlabel("Radial length (m)")
        ax.set_ylabel("Diffusion Coefficient (m^2 * s^-1)")
        ax.set_title(f"Radial Diffusion for {species}")

        def init():
            line.set_data([], [])
            return line,

        def update(frame_index):
            y = self.frame_diffusion_array(frame_index, spline)
            line.set_data(self.x, y)
            ax.set_title(f"Radial Diffusion for {species} at time {frame_index*0.5} us")
            return line,

        ani = FuncAnimation(fig, update, frames=num_frames,
                            init_func=init, interval=interval, blit=True)

        if save_path:
            if save_path.endswith(".gif"):
                from matplotlib.animation import PillowWriter
                ani.save(save_path, writer=PillowWriter(fps=1000//interval))
            else:
                from matplotlib.animation import FFMpegWriter
                ani.save(save_path, writer=FFMpegWriter(fps=1000//interval))

        plt.show()

    def get_data(self,frame_index):
        """
        Helper function to place all of the radial density averages
        in a single array. 

        Inputs: 
            - frame_index: Index of frame in question

        Outputs: 
            - array of average densities

        """

        dens_array = self.fp.load_data_frame("imp_density", frame=frame_index)
        density_array=np.zeros(len(self.x))

        for i in range(len(self.x)):
            density_array[i]=self.rAvg_density(i,dens_array)


        return density_array


    def data_fit_20_avg(self):
        """
        Helper function meant to average the density measurements
        from the last 20 frames of a simulation. Hopefully this gives
        a little more realistic results since the simulation should
        be in steady state by then!

        Inputs: None

        Outputs:
            - an array of averaged radial densities over the last 20 frames.


        """
        density_averages=np.zeros(len(self.x))
    
        # Go through each radial index
        for i in range(len(self.x)):

            tot = 0

            # For each radial index, average the poloidally averaged 
            # data from the last 20 frames

            for k in range(20):
                tot+=self.get_data(380+k)[i]

            density_averages[i]=tot/20


        return density_averages

    

    def lnIt(self, arr):
        """
        Helper function to return the same array, where 
        every entry is now its own natural log
        """
        return_me=np.zeros(len(arr))

        for i in range(len(arr)):
            if arr[i]!=0:
                return_me[i]=np.log(arr[i])
        return return_me




    def data_fit(self, plot="linear", nanbu_fitted=False):

        """
        This function will return the argument of the 
        exponential fit that best approximates the 
        diffusion coefficient data

        Inputs: 
            - frame_index: Frame index used for the array of
                diffusion coefficients

        Outputs: 
            - Some constant which corresponds to on average
                how the diffusion is exponentially decaying. 
                
        """


        def lin_fit(x,m,b):
            return m*x+b

        # Decide if you want to use the frame averaged data instead

        raw_y = self.lnIt(self.data_fit_20_avg())

        # Initialize arrays t
        cut_y = []
        cut_x = []


        # Fill in the arrays 
        # Fit to collisional or collisionless sims 

        if nanbu_fitted==True:
            n=2.31
        else:
            n=2.32

        # Helper function to restrict the inward transport data

        def cut_it(long_array):

            short_array=[]

            for i in range(len(self.x)):
                if n<self.x[i]<2.34:
                    short_array.append(long_array[i])

            return np.array(short_array)

        # Do this once for the x-values. 

        for i in range(len(self.x)):
                if n<self.x[i]<2.34:
                    cut_x.append(self.x[i]-2.259)



        # Convert lists to np arrays

        cut_y = np.array(cut_it(raw_y))
        cut_x = np.array(cut_x)
        popt,_ = curve_fit(lin_fit, cut_x, cut_y)
        m, b =popt


        if plot=="linear":
            plt.figure(figsize=(8, 6))

            # Plot of the actual data
            plt.plot(cut_x, cut_y,                   
                     label='Fit',
                     color='royalblue',
                     linewidth=4.5)

            # Scatter points for the fit function
            plt.scatter(cut_x, lin_fit(cut_x, *popt), 
                        label='Data', 
                        color='darkorange', 
                        edgecolors='black', 
                        linewidths=0.5, 
                        s=60, 
                        alpha=0.85)

            # Some font parameters
            plt.xlabel(r"$R - R_{\text{sep}}$ [m]", fontsize=20)
            plt.ylabel(r"ln(n$_W$) arb", fontsize=20)

            # Remove top and right spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Tick labels and the sort
            plt.tick_params(axis='both', which='major', labelsize=16, direction='in', length=6, width=1.2)
            plt.tick_params(axis='both', which='minor', labelsize=14, direction='in', length=3, width=1)

            plt.tight_layout()
            plt.show()
        
        elif plot =="exp":
        
            plt.figure(figsize=(8, 6))
            
            # Exponential fit to get 1/e decay length for indicated region
            def get_decay_length(x, xmin, xmax, y):
                
                # Mask data
                mask = np.logical_and(x >= xmin, x <= xmax)
                x = x[mask]
                y = y[mask]

                # Return error if zero in y data
                if (0 in y): print("Error! Can't fit if y contains zeros")

                ln_y=self.lnIt(y)
                
                # Linear regression ln(y) = bx + ln(a)
                slope, intercept, r_value, p_value, std_err = linregress(x, ln_y)

                # Go back to exponential
                a_fit = np.exp(intercept)    
                b_fit = slope

                # Return decay length and fit values
                return x, a_fit * np.exp(b_fit * x)


            # Data plot
            plt.plot(cut_x, cut_it(self.data_fit_20_avg()),                   
                     label='Data',
                     color='royalblue',
                     linewidth=4.5)

            # Fit plot
            plt.plot(get_decay_length(self.x, 0.05+2.259, 0.08+2.259, self.data_fit_20_avg()),
                        linestyle='--',
                        label='Fit',
                        color='darkorange')

            # Label parameters
            plt.xlabel(r"$R - R_{\text{sep}}$ [m]", fontsize=20)
            plt.ylabel(r"ln(n$_W$) arb", fontsize=20)

            # Remove top and right spines again
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Tick params. Change as necessary, but these work fine. 
            plt.tick_params(axis='both', which='major', labelsize=16, direction='in', length=6, width=1.2)
            plt.tick_params(axis='both', which='minor', labelsize=14, direction='in', length=3, width=1)

            plt.tight_layout()
            plt.show()
        
        else:
            return 1/m


        return 1/m

class MoneyPlot:
    """
    Here it is! This is the plot that you've all been waiting for. 

    This wall call instances of DiffusionPlot, and pull the argument 
    of the curve fit for various different instances of DiffusionPlot 
    based on the species. Then, this will be plotted against the Z 
    number for that impurity! 

    """

    def __init__(self,lithiumRun_n, boronRun, boronRun_n, carbonRun, carbonRun_n, nitrogenRun, nitrogenRun_n, oxygenRun_n,  neonRun, neonRun_n, ironRun, ironRun_n, molybdenumRun, molybdenumRun_n, xenonRun, xenonRun_n, tungstenRun, tungstenRun_n):
        self.mp_lithium_n = DiffusionPlot(lithiumRun_n)

        self.mp_boron_nc=DiffusionPlot(boronRun)
        self.mp_boron_n=DiffusionPlot(boronRun_n)

        self.mp_carbon_nc=DiffusionPlot(carbonRun)
        self.mp_carbon_n=DiffusionPlot(carbonRun_n)

        self.mp_nitrogen_n=DiffusionPlot(nitrogenRun_n)
        self.mp_nitrogen_nc=DiffusionPlot(nitrogenRun)

        self.mp_oxygen_n=DiffusionPlot(oxygenRun_n)

        self.mp_neon_nc=DiffusionPlot(neonRun)
        self.mp_neon_n=DiffusionPlot(neonRun_n)
        
        self.mp_iron_nc=DiffusionPlot(ironRun)
        self.mp_iron_n=DiffusionPlot(ironRun_n)

        self.mp_molybdenum_nc=DiffusionPlot(molybdenumRun)
        self.mp_molybdenum_n=DiffusionPlot(molybdenumRun_n)

        self.mp_xenon_nc=DiffusionPlot(xenonRun)
        self.mp_xenon_n=DiffusionPlot(xenonRun_n)

        self.mp_tungsten_nc=DiffusionPlot(tungstenRun)
        self.mp_tungsten_n=DiffusionPlot(tungstenRun_n)

        self.fp = flan_plots.FlanPlots(tungstenRun)
        self.x,_,_=self.fp.load_cell_centers()

    def filtered_points(self, arrX, arrY):
        """
        Helper function to create pairs of x and y points, thus
        omitting any points on the plot for which we don't have 
        impurity simulations. 

        Inputs: 
            - arrX : Array of x values
            - arrY : Array of y values

        Outputs: 
            - Two arrays, first one is the filtered x values, 
                second is the filtered Y values. 

        """
        #First calculate number of non-zero impurity entries

        def nonZero(arr):
            c=0
            for k in range(len(arr)):
                if arr[k]!=0:
                    c+=1
            return c

        # Initialize an array of the proper size 

        coordsY=np.zeros(nonZero(arr=arrY))
        coordsX=np.zeros(nonZero(arr=arrY))
        count = 0

        # Run through x values, pick up non-zero Y values 
        # and add them to coords array. 

        for i in range(len(arrX)):
            
            if arrY[i] != 0:
                coordsY[count] = arrY[i]
                coordsX[count] = arrX[i]
                count +=1

        return coordsX, coordsY




    def plot_lambda_charge(self, frame_index, nanbu_fitting=True):
        """
        This will plot the lambda vs. Z number

        Inputs: 
            - Frame_index: Needed for curve fit function. 
           
        Outputs: 
            - Plot of Z vs. lambda
        """

        # Initialize the arrays to be used. Just up to Z=74 for now. 
        z_array = np.arange(0,75)
        lambda_array_nc = np.zeros(75)
        lambda_array_n = np.zeros(75)
        
        # Set values for the non-collisional inward transport coefficient sims
        lambda_array_nc[4] = self.mp_boron_nc.data_fit(  "linear", nanbu_fitting)
        lambda_array_nc[5] = self.mp_carbon_nc.data_fit(  "linear",nanbu_fitting )
        lambda_array_nc[6] = self.mp_nitrogen_nc.data_fit( "linear", nanbu_fitting)
        lambda_array_nc[9] = self.mp_neon_nc.data_fit( "linear", nanbu_fitting)
        lambda_array_nc[25] = self.mp_iron_nc.data_fit(  "linear", nanbu_fitting)
        lambda_array_nc[41] = self.mp_molybdenum_nc.data_fit(  "linear", nanbu_fitting)
        lambda_array_nc[53] = self.mp_xenon_nc.data_fit(  "linear", nanbu_fitting)
        lambda_array_nc[73] = self.mp_tungsten_nc.data_fit(  "linear", nanbu_fitting) 
        


        # Collisional (n=nanbu) values for the various impurities. 
        lambda_array_n[2] = self.mp_lithium_n.data_fit(  "linear", nanbu_fitting)
        lambda_array_n[4] = self.mp_boron_n.data_fit(  "linear", nanbu_fitting)
        lambda_array_n[5] = self.mp_carbon_n.data_fit(  "linear", nanbu_fitting)
        lambda_array_n[6] = self.mp_nitrogen_n.data_fit(  "linear", nanbu_fitting)
        lambda_array_n[7] = self.mp_oxygen_n.data_fit( "linear", nanbu_fitting)
        lambda_array_n[9] = self.mp_neon_n.data_fit(  "linear", nanbu_fitting)
        lambda_array_n[25] = self.mp_iron_n.data_fit(  "linear", nanbu_fitting)
        lambda_array_n[41] = self.mp_molybdenum_n.data_fit(  "linear", nanbu_fitting)
        lambda_array_n[53] = self.mp_xenon_n.data_fit(  "linear", nanbu_fitting)
        lambda_array_n[73] = self.mp_tungsten_n.data_fit( "linear", nanbu_fitting)


        # Take the non-zero inward transport coefficients
        filteredZ_n, _ = self.filtered_points(z_array, lambda_array_n)
        filteredZ_nc, _ = self.filtered_points(z_array, lambda_array_nc)
        _, filteredY_nc = self.filtered_points(z_array,lambda_array_nc)
        _, filteredY_n = self.filtered_points(z_array, lambda_array_n)
        
        # Plot those jokers
        plt.plot(filteredZ_nc, filteredY_nc, label = 'Collisionless', color='blue')
        plt.plot(filteredZ_n, filteredY_n, label = 'Collisional', color = 'red')

        # Give it a name
        plt.title('Log of Density fit argument vs. impurity Z number')
        plt.xlabel("Impurity Z number")
        plt.ylabel("Exponential argument lambda=ln(1/b)")
        


        plt.legend()

        plt.show()

        return filteredZ_n, filteredY_nc, filteredY_n
        

    
class ChargePlot(): 
    """
    
    This class plots the second deliverable for the SULI project ; which probes 
    the density profiles of simulations that start with different charge states.


    """

    def __init__(self, c1, c2, c3, c4, c5, c6, c7):
        self.cp_c1=DiffusionPlot(c1)
        self.cp_c2=DiffusionPlot(c2)
        self.cp_c3=DiffusionPlot(c3)
        self.cp_c4=DiffusionPlot(c4)
        self.cp_c5=DiffusionPlot(c5)
        self.cp_c6=DiffusionPlot(c6)
        self.cp_c7=DiffusionPlot(c7)
        
        self.fp = flan_plots.FlanPlots(c7)
        self.x,_,_ =self.fp.load_cell_centers()



    def density_profiles(self, obj, charge):
        """
        Helper function to add a density profile to a plot
        """

        plt.plot(self.x, obj.data_fit_20_avg(), label=f"Charge {charge}")




    def plot_charge_density(self):
        """
        This function plots the poloidally averaged density profile 
        (averaged over the last 20 frames of a 400 frame simulation, 
        supposedly where we've reached some kind of equilibrium) vs. 
        the charge of the simulation for which the data was acquired. 

        """

       # Function to get density data for each object and put it 
       # in the plot 

        all_densities=[]
        for i in range(1, 8):
            obj = getattr(self, f"cp_c{i}")
            all_densities.append(obj.data_fit_20_avg())
            self.density_profiles(obj, i)

        plt.xlabel(r"R-R$_{\{text{sep}}$")
        plt.ylabel("Average density")
        plt.legend()
        plt.show()

        return all_densities




