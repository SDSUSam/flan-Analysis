import numpy as np
from matplotlib import patheffects
from scipy.stats import linregress
import sympy as sp
import matplotlib.pyplot as plt
import flan_plots
from scipy.optimize import curve_fit

class Fits:

    def __init__(self):
        self.data = np.array([7.78481277e-07, 1.01823624e-06, 1.20420471e-06, 1.40876147e-06,
       1.59214829e-06, 1.68402815e-06, 1.87927476e-06, 2.07237694e-06,
       2.15472590e-06, 2.29975519e-06, 2.44819103e-06, 2.43162546e-06,
       2.63401242e-06, 2.70344843e-06, 2.62957605e-06, 2.72346194e-06,
       2.80958207e-06, 2.98931624e-06, 3.08803916e-06, 3.15210514e-06,
       3.33487539e-06, 3.64541882e-06, 3.96930629e-06, 4.20508113e-06,
       4.76160258e-06, 5.21439104e-06, 5.90137255e-06, 6.80642172e-06,
       7.68058847e-06, 8.44574109e-06, 9.24027874e-06, 1.07245003e-05,
       1.20037186e-05, 1.25889387e-05, 1.39428284e-05, 1.47768363e-05,
       1.52040377e-05, 1.56455239e-05, 1.60827557e-05, 1.67084271e-05,
       1.73073498e-05, 1.73555030e-05, 1.82160268e-05, 1.92218604e-05,
       2.01949713e-05, 2.17974686e-05, 2.32718048e-05, 2.55290204e-05,
       2.86368910e-05, 3.23227906e-05, 3.56651422e-05, 4.00707372e-05,
       4.43243935e-05, 4.83300967e-05, 5.14329821e-05, 5.41485427e-05,
       5.62134357e-05, 5.83456065e-05, 6.21708122e-05, 6.51185582e-05,
       6.97542602e-05, 7.55825726e-05, 8.26044546e-05, 9.22343169e-05,
       1.01939512e-04, 1.13057835e-04, 1.22089894e-04, 1.31187146e-04,
       1.40783350e-04, 1.49223308e-04, 1.60614353e-04, 1.71108960e-04,
       1.85462216e-04, 2.00197839e-04, 2.19138455e-04, 2.41765742e-04,
       2.64459424e-04, 2.90422235e-04, 3.14860669e-04, 3.36613409e-04,
       3.55920969e-04, 3.82353578e-04, 4.20748833e-04, 4.72127360e-04,
       5.52872045e-04, 6.67917563e-04, 8.19214952e-04, 1.01545446e-03,
       1.25860915e-03, 1.59162678e-03, 2.39789022e-03, 4.29059825e-03,
       2.67817498e-03, 1.79924741e-03, 1.19584419e-03, 6.16494349e-04])
        self.x = np.array([0.0412872 , 0.04186172, 0.04243624, 0.04301076, 0.04358528,
       0.04415979, 0.04473431, 0.04530883, 0.04588335, 0.04645787,
       0.04703239, 0.04760691, 0.04818143, 0.04875595, 0.04933047,
       0.04990498, 0.0504795 , 0.05105402, 0.05162854, 0.05220306,
       0.05277758, 0.0533521 , 0.05392662, 0.05450114, 0.05507565,
       0.05565017, 0.05622469, 0.05679921, 0.05737373, 0.05794825,
       0.05852277, 0.05909729, 0.05967181, 0.06024633, 0.06082084,
       0.06139536, 0.06196988, 0.0625444 , 0.06311892, 0.06369344,
       0.06426796, 0.06484248, 0.065417  , 0.06599151, 0.06656603,
       0.06714055, 0.06771507, 0.06828959, 0.06886411, 0.06943863,
       0.07001315, 0.07058767, 0.07116219, 0.0717367 , 0.07231122,
       0.07288574, 0.07346026, 0.07403478, 0.0746093 , 0.07518382,
       0.07575834, 0.07633286, 0.07690737, 0.07748189, 0.07805641,
       0.07863093, 0.07920545, 0.07977997, 0.08035449, 0.08092901,
       0.08150353, 0.08207805, 0.08265256, 0.08322708, 0.0838016 ,
       0.08437612, 0.08495064, 0.08552516, 0.08609968, 0.0866742 ,
       0.08724872, 0.08782323, 0.08839775, 0.08897227, 0.08954679,
       0.09012131, 0.09069583, 0.09127035, 0.09184487, 0.09241939,
       0.09299391, 0.09356842, 0.09414294, 0.09471746, 0.09529198,
       0.0958665 ])

    def return_data(self):
        return self.data


    def get_decay_length(self, x, xmin, xmax, y):

        # Mask data
        mask = np.logical_and(x > xmin, x < xmax)
        x = x[mask]
        y = y[mask]

        # Return error if zero in y data
        if (0 in y): print("Error! Can't fit if y contains zeros")

        # Natural log of data
        ln_y=np.log(y)

        # Linear regression ln(y) = bx + ln(a)
        slope, intercept, r_value, p_value, std_err = linregress(x, ln_y)

        # Go back to exponential
        a_fit = np.exp(intercept)
        b_fit = slope

        # Return decay length and fit values
        return x, a_fit * np.exp(b_fit * x)


    def plot_us(self, xmin, xmax):
        """
        Plotting function
        """
        # Mask and slice data
        mask = np.logical_and(self.x > xmin, self.x < xmax)
        x_data = self.x[mask]
        y_data = self.data[mask]

        # Fit data
        _, fit = self.get_decay_length(self.x, xmin, xmax, self.data)

        # Define path effects for black outline
        outline = [patheffects.withStroke(linewidth=8, foreground="black")]

        # Plot Data - royal blue with black outline
        plt.plot(
            100*x_data, y_data,
            label='Density',
            color='royalblue',
            linewidth=6.5,
            path_effects=outline
        )


        # Plot Fit - orange with black outline
        """
        plt.plot(
            100*x_data, fit,
            label='Fit',
            color='darkorange',
            linestyle='--',
            linewidth=6,
            path_effects=outline
        )
        """

        # Spine stuff

        ax=plt.gca()

        for spine in ('right', 'top'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        plt.tick_params(axis='both', which='major', labelsize=14)

        # Make it look cleaner
        plt.xlabel(r"R-R$_{\text{sep}}$ (cm)", fontsize=18)
        plt.ylabel(r"$n_W \cdot \text{arb}$", fontsize=18)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.show()
