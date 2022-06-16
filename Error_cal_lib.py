"""
Original code by
Copyright (C) 2022  Shubin Zhang
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy as sp
import scipy.optimize
import math
import csv
class simulation():
    def __init__(self, response = None, calibration = None, L = 0, verbose = False, plot = False):
        wl_cmf, x_cmf, y_cmf ,z_cmf = np.loadtxt("Example\\color_matching_function.csv",  delimiter = ",", unpack = True)
        wl_min, wl_max = min(wl_cmf), max(wl_cmf)
        x_function = interp1d(wl_cmf, x_cmf, kind = "cubic")
        y_function = interp1d(wl_cmf, y_cmf, kind = "cubic")
        z_function = interp1d(wl_cmf, z_cmf, kind = "cubic")
        self.wl = np.linspace(wl_min, wl_max, 801)
        x_cmf = x_function(self.wl)
        y_cmf = y_function(self.wl)
        z_cmf = z_function(self.wl)
        N_factor = max([max(x_cmf),max(y_cmf),max(z_cmf)])
        self.x_cmf = x_cmf/N_factor
        self.y_cmf = y_cmf/N_factor
        self.z_cmf = z_cmf/N_factor
        self.verbose = verbose
        if response:
            self.load_response(response)
        else:
            self.load_response()
        if L == 0:
            self.load_calibration(calibration)
        else:
            self.load_calibration(calibration, L)
        if plot:
            plt.figure()
            plt.plot(self.wl, self.x_response, "r", label = "x")
            plt.plot(self.wl, self.y_response, "g", label = "y")
            plt.plot(self.wl, self.z_response, "b", label = "z")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Instrument Response (a.u.)")
            plt.legend(loc="upper left")
            plt.title("Colorimeter Spectral Respone")
            plt.figure()
            plt.plot(self.wl, self.x_ls, "r", label = "Red")
            plt.plot(self.wl, self.y_ls, "g", label = "Green")
            plt.plot(self.wl, self.z_ls, "b", label = "Blue")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Intensity (a.u.)")
            plt.legend(loc="upper left")
            plt.title("Calibration Light Source Spectrum")
            plt.show()

    def load_response(self,filename = "Example\\RGB_response.csv"):
        #Load filter&camera spectral response 
        wl_response, x_response, y_response ,z_response = np.loadtxt(filename,  delimiter = ",", unpack = True)
        x_function = interp1d(wl_response, x_response, bounds_error = False, fill_value = 0 )
        y_function = interp1d(wl_response, y_response, bounds_error = False, fill_value = 0 )
        z_function = interp1d(wl_response, z_response, bounds_error = False, fill_value = 0 )
        x_response = x_function(self.wl)
        y_response = y_function(self.wl)
        z_response = z_function(self.wl)
        N_factor = max([max(x_response),max(y_response),max(z_response)])
        self.x_response = x_response/N_factor
        self.y_response = y_response/N_factor
        self.z_response = z_response/N_factor
        

    def load_dut_spectrum(self, filename = None):
        #Load DUT spectrum
        wl, x_dut, y_dut, z_dut = np.loadtxt(filename,  delimiter = ",", unpack = True)
        x_function = interp1d(wl, x_dut, bounds_error = False, fill_value = 0 )
        y_function = interp1d(wl, y_dut, bounds_error = False, fill_value = 0 )
        z_function = interp1d(wl, z_dut, bounds_error = False, fill_value = 0 )
        return x_function(self.wl), y_function(self.wl), z_function(self.wl)

    def generate_guass_spectrum(self, dict):
        return self.gauss(dict["x_cw"], dict["x_fwhm"]), self.gauss(dict["y_cw"], dict["y_fwhm"]), self.gauss(dict["z_cw"], dict["z_fwhm"])

    def gauss(self, cw,fwhm):
        sigma  = fwhm/(2*math.sqrt(2*math.log(2)))
        return 1/(sigma*np.sqrt(2*math.pi))*np.exp(-(self.wl-cw)**2/(2*sigma**2))

    def load_calibration(self, arg = None, L = 57030):
        #Load calibration light source primary color spectra
        #arg: Calibration light source spectrum file name
        #L (nit): Red color luminance reading from light source 
        if not arg:
            #Load primary color spectra from Labsphere Tunable RGB Light Source (CSTM-CCS-RGB-REF)
            xwl_ls, x_ls, y_ls, z_ls = np.loadtxt("Example\\Calibration_LS.csv",  delimiter = ",", unpack = True)
        elif len(arg) == 1:
            xwl_ls, x_ls, y_ls, z_ls = np.loadtxt(arg,  delimiter = ",", unpack = True)
        else:
            print("can't input calibration data")
        x_function = interp1d(xwl_ls, x_ls, bounds_error = False, fill_value = 0 )
        y_function = interp1d(xwl_ls, y_ls, bounds_error = False, fill_value = 0 )
        z_function = interp1d(xwl_ls, z_ls, bounds_error = False, fill_value = 0 )
        self.x_ls = x_function(self.wl)
        self.y_ls = y_function(self.wl)
        self.z_ls = z_function(self.wl)
        self.X_R = sum(self.x_ls*self.x_cmf)
        self.Y_R = sum(self.x_ls*self.y_cmf)
        self.Z_R = sum(self.x_ls*self.z_cmf)
        L_factor = L/self.Y_R
        self.x_ls, self.y_ls, self.z_ls = self.x_ls*L_factor, self.y_ls*L_factor, self.z_ls*L_factor
        self.w_ls = self.x_ls+self.y_ls+self.z_ls
        _, self.Y_R, _, self.C_xR, self.C_yR = self.calculate_coordinate_CIE(self.x_ls)
        _, self.Y_G, _, self.C_xG, self.C_yG = self.calculate_coordinate_CIE(self.y_ls)
        _, self.Y_B, _, self.C_xB, self.C_yB = self.calculate_coordinate_CIE(self.z_ls)
        _, self.Y_W, _, self.C_xW, self.C_yW = self.calculate_coordinate_CIE(self.w_ls)


    def calculate_coordinate_CIE(self, x):
        #Calculate Tristimulus/Chromaticity based on CIE color matching function
        X = sum(x*self.x_cmf)
        Y = sum(x*self.y_cmf)
        Z = sum(x*self.z_cmf)
        C_x = X/(X+Y+Z)
        C_y = Y/(X+Y+Z)
        if self.verbose:
            print("CIE")
            print("Cx=",C_x)
            print("Cy=",C_y)
            print("L=",Y)
        return X, Y, Z, C_x, C_y
    
    def calculate_coordinate_CM(self, x, CIE_correction = [1,1,1]):
        #Calculate Tristimulus/Chromaticity based on colormatrix generated from four color calibration
        x_raw = sum(x*self.x_response)
        y_raw = sum(x*self.y_response)
        z_raw = sum(x*self.z_response)
        Tri_vector = np.dot(self.colormatrix, [x_raw,y_raw,z_raw])
        X, Y, Z = Tri_vector[0]/CIE_correction[0], Tri_vector[1]/CIE_correction[1], Tri_vector[2]/CIE_correction[2]
        C_x = X/(X+Y+Z)
        C_y = Y/(X+Y+Z)
        if self.verbose:
            print("colormatrix")
            print("Cx=",C_x)
            print("Cy=",C_y)
            print("L=",Y)
        return X, Y, Z, C_x, C_y


    def generate_colormatrix(self, guess_matrix = None, verbose = False):
        #guess: 2d color matrix
        if not guess_matrix:
            guess = np.array([0.01,0.01,0.01,-0.01,0.01,0.01,0.00001,0.01,-0.01,self.Y_R,self.Y_G,self.Y_B])
        else:
            guess = guess_matrix.flatten()
            np.append(guess, [self.Y_R,self.Y_G,self.Y_B])
        response_list = [self.x_response, self.y_response, self.z_response]
        Spectrum_list = [self.w_ls, self.x_ls, self.y_ls, self.z_ls]
        raw = np.zeros((4,3))
        color_array = np.array([[self.C_xW,self.C_yW],[self.C_xR,self.C_yR],[self.C_xG,self.C_yG],[self.C_xB,self.C_yB]])
        L = self.Y_W
        for i in range(4):
            for j in range(3):
                raw[i,j] = sum(Spectrum_list[i]*response_list[j])
        def colormatrix(guess):
            return [
                guess[0]*raw[0,0]+guess[1]*raw[0,1]+guess[2]*raw[0,2]-L*color_array[0,0]/color_array[0,1],
                guess[3]*raw[0,0]+guess[4]*raw[0,1]+guess[5]*raw[0,2]-L,
                guess[6]*raw[0,0]+guess[7]*raw[0,1]+guess[8]*raw[0,2]-L*(1-color_array[0,0]-color_array[0,1])/color_array[0,1],
                guess[0]*raw[1,0]+guess[1]*raw[1,1]+guess[2]*raw[1,2]-guess[9]*color_array[1,0]/color_array[1,1],
                guess[3]*raw[1,0]+guess[4]*raw[1,1]+guess[5]*raw[1,2]-guess[9],
                guess[6]*raw[1,0]+guess[7]*raw[1,1]+guess[8]*raw[1,2]-guess[9]*(1-color_array[1,0]-color_array[1,1])/color_array[1,1],
                guess[0]*raw[2,0]+guess[1]*raw[2,1]+guess[2]*raw[2,2]-guess[10]*color_array[2,0]/color_array[2,1],
                guess[3]*raw[2,0]+guess[4]*raw[2,1]+guess[5]*raw[2,2]-guess[10],
                guess[6]*raw[2,0]+guess[7]*raw[2,1]+guess[8]*raw[2,2]-guess[10]*(1-color_array[2,0]-color_array[2,1])/color_array[2,1],
                guess[0]*raw[3,0]+guess[1]*raw[3,1]+guess[2]*raw[3,2]-guess[11]*color_array[3,0]/color_array[3,1],
                guess[3]*raw[3,0]+guess[4]*raw[3,1]+guess[5]*raw[3,2]-guess[11],
                guess[6]*raw[3,0]+guess[7]*raw[3,1]+guess[8]*raw[3,2]-guess[11]*(1-color_array[3,0]-color_array[3,1])/color_array[3,1],   
                ]

        sol =sp.optimize.root(colormatrix, guess, method = "hybr")["x"][:9]
        self.colormatrix = np.reshape(sol,(3,3))
        if verbose:
            print("Colormatrix")
            print(self.colormatrix)

    
    

    
    def error_calculation(self, mode = "shift", output = "lab", kwargs = {} ):
        """
        mode: "shift" calculate error from spectral shift between calibration and DUT spectrum (assume identical shape)
              "load" load dut spectrum and calculate error
              "gauss" Assume DUT spectrum follows Gaussian distribution. Calculate error based on provided DUT spectrum central wavelength&FWHM  
        """     
        if mode == "shift":
            shift = int(kwargs["shift"]*2)
            
            x = np.roll(self.x_ls, shift)
            y = np.roll(self.y_ls, shift)
            z = np.roll(self.z_ls, shift)
        elif mode == "load":
            x, y, z = self.load_dut_spectrum(filename = kwargs["filename"])

        elif mode == "gauss":
            x, y, z = self.generate_guass_spectrum(dict = kwargs)
        else:
            print("Incorrect error calculation mode")
            return


        #Ground truth from color matching function

        #Red
        if self.verbose:
            print("CIE_Red")
        X_R_CIE, Y_R_CIE, Z_R_CIE, C_xR_CIE, C_yR_CIE = self.calculate_coordinate_CIE(x)
        #Green
        if self.verbose:
            print("CIE_Green")
        X_G_CIE, Y_G_CIE, Z_G_CIE, C_xG_CIE, C_yG_CIE = self.calculate_coordinate_CIE(y)
        #Blue
        if self.verbose:
            print("CIE_Blue")
        X_B_CIE, Y_B_CIE, Z_B_CIE, C_xB_CIE, C_yB_CIE = self.calculate_coordinate_CIE(z)
        
        #calculate u' v'
        uprime_R_CIE = 4*X_R_CIE/(X_R_CIE + 15*Y_R_CIE + 3*Z_R_CIE)
        vprime_R_CIE = 9*Y_R_CIE/(X_R_CIE + 15*Y_R_CIE + 3*Z_R_CIE)
        uprime_G_CIE = 4*X_G_CIE/(X_G_CIE + 15*Y_G_CIE + 3*Z_G_CIE)
        vprime_G_CIE = 9*Y_G_CIE/(X_G_CIE + 15*Y_G_CIE + 3*Z_G_CIE)
        uprime_B_CIE = 4*X_B_CIE/(X_B_CIE + 15*Y_B_CIE + 3*Z_B_CIE)
        vprime_B_CIE = 9*Y_B_CIE/(X_B_CIE + 15*Y_B_CIE + 3*Z_B_CIE)


        if self.verbose:
            print("CM_Red")
        X_R_CM, Y_R_CM, Z_R_CM, C_xR_CM, C_yR_CM = self.calculate_coordinate_CM(x)

        if self.verbose:
            print("CM_Green")
        X_G_CM, Y_G_CM, Z_G_CM, C_xG_CM, C_yG_CM = self.calculate_coordinate_CM(y)

        if self.verbose:
            print("CM_Blue")
        X_B_CM, Y_B_CM, Z_B_CM, C_xB_CM, C_yB_CM = self.calculate_coordinate_CM(z)

        uprime_R_CM = 4*X_R_CM/(X_R_CM + 15*Y_R_CM + 3*Z_R_CM)
        vprime_R_CM = 9*Y_R_CM/(X_R_CM + 15*Y_R_CM + 3*Z_R_CM)
        uprime_G_CM = 4*X_G_CM/(X_G_CM + 15*Y_G_CM + 3*Z_G_CM)
        vprime_G_CM = 9*Y_G_CM/(X_G_CM + 15*Y_G_CM + 3*Z_G_CM)
        uprime_B_CM = 4*X_B_CM/(X_B_CM + 15*Y_B_CM + 3*Z_B_CM)
        vprime_B_CM = 9*Y_B_CM/(X_B_CM + 15*Y_B_CM + 3*Z_B_CM)
        

        
        if output == "xyz":
            #output luminance/chromaticity
            return [Y_R_CIE, C_xR_CIE, C_yR_CIE, Y_G_CIE, C_xG_CIE, C_yG_CIE, Y_B_CIE, C_xB_CIE, C_yB_CIE], [Y_R_CM, C_xR_CM, C_yR_CM, Y_G_CM, C_xG_CM, C_yG_CM, Y_B_CM, C_xB_CM, C_yB_CM], [[uprime_R_CIE, vprime_R_CIE, uprime_G_CIE, vprime_G_CIE, uprime_B_CIE, vprime_B_CIE], [uprime_R_CM, vprime_R_CM, uprime_G_CM, vprime_G_CM, uprime_B_CM, vprime_B_CM]]
        elif output == "lab":
            #output L*a*b and color difference delta E
            L_R_CIE, a_R_CIE, b_R_CIE = self.xyz2lab(C_xR_CIE, C_yR_CIE, 1 - C_xR_CIE - C_yR_CIE)
            L_G_CIE, a_G_CIE, b_G_CIE = self.xyz2lab(C_xG_CIE, C_yG_CIE, 1 - C_xG_CIE - C_yG_CIE)
            L_B_CIE, a_B_CIE, b_B_CIE = self.xyz2lab(C_xB_CIE, C_yB_CIE, 1 - C_xB_CIE - C_yB_CIE)
            L_R_CM, a_R_CM, b_R_CM = self.xyz2lab(C_xR_CM, C_yR_CM, 1 - C_xR_CM - C_yR_CM)
            L_G_CM, a_G_CM, b_G_CM = self.xyz2lab(C_xG_CM, C_yG_CM, 1 - C_xG_CM - C_yG_CM)
            L_B_CM, a_B_CM, b_B_CM = self.xyz2lab(C_xB_CM, C_yB_CM, 1 - C_xB_CM - C_yB_CM)


            deltaE_R = math.sqrt((L_R_CIE - L_R_CM)**2 + (a_R_CIE - a_R_CM)**2 + (b_R_CIE - b_R_CM)**2)
            deltaE_G = math.sqrt((L_G_CIE - L_G_CM)**2 + (a_G_CIE - a_G_CM)**2 + (b_G_CIE - b_G_CM)**2)
            deltaE_B = math.sqrt((L_B_CIE - L_B_CM)**2 + (a_B_CIE - a_B_CM)**2 + (b_B_CIE - b_B_CM)**2)

            return [L_R_CIE, a_R_CIE, b_R_CIE, L_G_CIE, a_G_CIE, b_G_CIE, L_B_CIE, a_B_CIE, b_B_CIE], [L_R_CM, a_R_CM, b_R_CM, L_G_CM, a_G_CM, b_G_CM, L_B_CM, a_B_CM, b_B_CM], [deltaE_R, deltaE_G, deltaE_B]
        else:
            print("invalid output mode")
    
    def xyz2lab(self, X, Y, Z):
        #calculate L*a*b from Tristimulus
        whitepoint = [95.05, 100, 108.88] # D65
        Xr, Yr, Zr = whitepoint[0], whitepoint[1], whitepoint[2]
        xr, yr, zr = X/Xr, Y/Yr, Z/Zr
        eps = 0.008856
        k = 903.3
        exp = 1/3
        f_list = [i**exp if i > eps else (k*i+16)/116 for i in [xr, yr, zr]]
        L = 116*f_list[1] - 16
        a = 500*(f_list[0] - f_list[1])
        b = 200*(f_list[1] - f_list[2])
        if self.verbose:
            print("L,\t\t,a,\t\t,b")
            print(L,a,b)
        return L, a, b    

    def output(self, cal_mode = "shift", input = None):
        if cal_mode == "shift":
            if input == None:
                input = [-10.0, 10.0, 0.5]
            try:
                num_of_step = int((input[1]-input[0])/input[2] + 1)
            except:
                print("invalid input")
                return
            with open("result.csv", "w", newline="",  encoding="utf-8") as csv_file:
                fieldnames = ["Spectral Shfit (nm)", "Red \u0394E", "Green \u0394E", "Blue \u0394E", "Red L (%)", "Green L (%)", "Blue L (%)", "Red \u0394u", "Red \u0394v", "Green \u0394u", "Green \u0394v", "Blue \u0394u", "Blue \u0394v"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                kwargs = {}
                for i in np.linspace(input[0], input[1], num_of_step):
                    kwargs["shift"]=i
                    a, b, c  = self.error_calculation(mode = "shift", output = "lab", kwargs = kwargs)
                    d, e, f = self.error_calculation(mode = "shift", output = "xyz", kwargs = kwargs)
                    L_ratio_R = 100*(d[0] - e[0])/d[0]
                    L_ratio_G = 100*(d[3] - e[3])/d[3]
                    L_ratio_B = 100*(d[6] - e[6])/d[6]
                    delta_u_R = abs(f[0][0] - f[1][0])
                    delta_v_R = abs(f[0][1] - f[1][1])
                    delta_u_G = abs(f[0][2] - f[1][2])
                    delta_v_G = abs(f[0][3] - f[1][3])
                    delta_u_B = abs(f[0][4] - f[1][4])
                    delta_v_B = abs(f[0][5] - f[1][5])
                    writer.writerow({"Spectral Shfit (nm)":i, "Red \u0394E":c[0], "Green \u0394E":c[1], "Blue \u0394E":c[2], "Red L (%)":L_ratio_R, "Green L (%)":L_ratio_G, "Blue L (%)":L_ratio_B, "Red \u0394u":delta_u_R, "Red \u0394v":delta_v_R, "Green \u0394u":delta_u_G, "Green \u0394v":delta_v_G, "Blue \u0394u":delta_u_B, "Blue \u0394v":delta_v_B})
                
        elif cal_mode == "load":
            with open("result.csv", "w", newline="",  encoding="utf-8") as csv_file:
                fieldnames = ["Spectral Shfit (nm)", "Red \u0394E", "Green \u0394E", "Blue \u0394E", "Red L (%)", "Green L (%)", "Blue L (%)", "Red \u0394u", "Red \u0394v", "Green \u0394u", "Green \u0394v", "Blue \u0394u", "Blue \u0394v"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                kwargs = {}  
                kwargs["filename"] = input
                a, b, c  = self.error_calculation(mode = "shift", output = "lab", kwargs = kwargs)
                d, e, f = self.error_calculation(mode = "shift", output = "xyz", kwargs = kwargs)
                L_ratio_R = 100*(d[0] - e[0])/d[0]
                L_ratio_G = 100*(d[3] - e[3])/d[3]
                L_ratio_B = 100*(d[6] - e[6])/d[6]
                delta_u_R = abs(f[0][0] - f[1][0])
                delta_v_R = abs(f[0][1] - f[1][1])
                delta_u_G = abs(f[0][2] - f[1][2])
                delta_v_G = abs(f[0][3] - f[1][3])
                delta_u_B = abs(f[0][4] - f[1][4])
                delta_v_B = abs(f[0][5] - f[1][5])
                writer.writerow({"File":input, "Red \u0394E":c[0], "Green \u0394E":c[1], "Blue \u0394E":c[2], "Red L (%)":L_ratio_R, "Green L (%)":L_ratio_G, "Blue L (%)":L_ratio_B, "Red \u0394u":delta_u_R, "Red \u0394v":delta_v_R, "Green \u0394u":delta_u_G, "Green \u0394v":delta_v_G, "Blue \u0394u":delta_u_B, "Blue \u0394v":delta_v_B})

        elif cal_mode == "gauss":
            data_dict = {}
            if not input:
                input = 'Example\\FWHM_and_central_wavelength.csv'
            with open(input, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    sn = row['filename'].strip().split(".")[0]
                    if sn[:-1] not in data_dict:
                        data_dict[sn[:-1]] = {}
                    if sn[-1] == "R":
                        data_dict[sn[:-1]]["x_cw"] = float(row["central wavelength - nm"])
                        data_dict[sn[:-1]]["x_fwhm"] = float(row["FWHM - nm"])
                        data_dict[sn[:-1]]["x_shift"] = float(row["central wavelength - nm"]) - self.wl[np.where(self.x_ls == max(self.x_ls))]
                    elif sn[-1] == "G":
                        data_dict[sn[:-1]]["y_cw"] = float(row["central wavelength - nm"])
                        data_dict[sn[:-1]]["y_fwhm"] = float(row["FWHM - nm"])
                        data_dict[sn[:-1]]["y_shift"] = float(row["central wavelength - nm"]) - self.wl[np.where(self.y_ls == max(self.y_ls))]
                    elif sn[-1] == "B":
                        data_dict[sn[:-1]]["z_cw"] = float(row["central wavelength - nm"])
                        data_dict[sn[:-1]]["z_fwhm"] = float(row["FWHM - nm"])
                        data_dict[sn[:-1]]["z_shift"] = float(row["central wavelength - nm"]) - self.wl[np.where(self.z_ls == max(self.z_ls))]
                    else:
                        print("invalid SN")

                with open("result.csv", "w", newline="",  encoding="utf-8") as csv_file:
                    fieldnames = ["SN", "Red Spectral Shfit (nm)", "Red \u0394E", "Green Spectral Shfit (nm)", "Green \u0394E", "Blue Spectral Shfit (nm)", "Blue \u0394E" , "Red L (%)", "Green L (%)", "Blue L (%)", "Red \u0394u", "Red \u0394v", "Green \u0394u", "Green \u0394v", "Blue \u0394u", "Blue \u0394v"]
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    for key, value in data_dict.items():
                        a, b, c  = self.error_calculation(mode = "gauss", output = "lab", kwargs = value)
                        d, e, f = self.error_calculation(mode = "gauss", output = "xyz", kwargs = value)
                        L_ratio_R = 100*(d[0] - e[0])/d[0]
                        L_ratio_G = 100*(d[3] - e[3])/d[3]
                        L_ratio_B = 100*(d[6] - e[6])/d[6]
                        delta_u_R = abs(f[0][0]-f[1][0])
                        delta_v_R = abs(f[0][1]-f[1][1])
                        delta_u_G = abs(f[0][2]-f[1][2])
                        delta_v_G = abs(f[0][3]-f[1][3])
                        delta_u_B = abs(f[0][4]-f[1][4])
                        delta_v_B = abs(f[0][5]-f[1][5])
                        writer.writerow({"SN":key, "Red Spectral Shfit (nm)": value["x_shift"][0], "Red \u0394E":c[0], "Green Spectral Shfit (nm)": value["y_shift"][0], "Green \u0394E":c[1], "Blue Spectral Shfit (nm)": value["z_shift"][0],"Blue \u0394E":c[2], "Red L (%)":L_ratio_R, "Green L (%)":L_ratio_G, "Blue L (%)":L_ratio_B, "Red \u0394u":delta_u_R, "Red \u0394v":delta_v_R, "Green \u0394u":delta_u_G, "Green \u0394v":delta_v_G, "Blue \u0394u":delta_u_B, "Blue \u0394v":delta_v_B})

        else:
            print("Invalid calculation mode")
            return
        print("Error Calculation Finished")

if __name__ == "__main__":
   pass