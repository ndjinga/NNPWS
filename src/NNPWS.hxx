/****************************************************************************
* Copyright (c) 2025, CEA
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
* OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*****************************************************************************/


#ifndef __NNPWS_HH__
#define __NNPWS_HH__

#pragma once

#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
using std::istringstream ;
using std::ostringstream ;
#include <typeinfo>
#include <string>
#include <iostream>
#include <vector>
#include <torch/script.h>
#include "ModelLoader.hxx"
#include "Regions.hxx"


//! NNPWS class
class NNPWS {
public:
    //! Constructor method
    NNPWS();

    //! Copy constructor
    NNPWS(const NNPWS &);

    //! Destructor method
    ~NNPWS();

    //! for initialisation (memory allocations) :
    int init();

    //! Getter methods
    /*
    const std::string& table_name()   const ;
    const std::string& version_name() const ;
    const std::string& phase_name()   const ;
    const std::string& fluid_name() const;
    const std::string& equation_name() const;
    */

    //! Global methods
    double compute_Ph(const char *const property_name, double in1, double in2) const;

    double compute_PT(const char *const property_name, double in1, double in2) const;

    double compute_Ph(const char *const property_name, int property_number, double in1, double in2) const;

    double compute_PT(const char *const property_name, int property_number, double in1, double in2) const;

    //Saturation methods
    // T_sat(P)
    double compute_T_sat_p(double p) const;

    double compute_p_sat_T(double T) const;

    // rho_lsat(P), rho_vsat(P)
    double compute_rho_l_sat_p(double p) const;

    double compute_rho_v_sat_p(double p) const;

    double compute_rho_l_sat_T(double T) const;

    double compute_rho_v_sat_T(double T) const;

    // h_lsat(P), h_vsat(P)
    double compute_h_l_sat_p(double p) const;

    double compute_h_v_sat_p(double p) const;

    // cp_lsat(P), cp_vsat(P)
    double compute_cp_l_sat_p(double p) const;

    double compute_cp_v_sat_p(double p) const;

    double compute_cp_l_sat_T(double T) const;

    double compute_cp_v_sat_T(double T) const;

    // h_lsat(T), h_vsat(T)
    double compute_h_l_sat_T(double T) const;

    double compute_h_v_sat_T(double T) const;

    //ph functions
    //! Pr(p,h)
    double compute_pr_ph(double p, double h) const;

    //! h(p,T)
    double compute_h_pT(double p, double T) const;

    double compute_T_ph(double p, double h) const;

    //! rho(p,T)
    double compute_rho_pT(double p, double T) const;

    double compute_rho_ph(double p, double h) const;

    //! u(p,T)
    double compute_u_pT(double p, double T) const;

    double compute_u_ph(double p, double h) const;

    //! s(p,T)
    double compute_s_pT(double p, double T) const;

    double compute_s_ph(double p, double h) const;

    //! mu(p,T)
    double compute_mu_pT(double p, double T) const;

    double compute_mu_ph(double p, double h) const;

    //! lambda(p,T)
    double compute_lambda_pT(double p, double T) const;

    double compute_lambda_ph(double p, double h) const;

    //! cp(p,T)
    double compute_cp_pT(double p, double T) const;

    double compute_cp_ph(double p, double h) const;

    //! cv(p,T)
    double compute_cv_pT(double p, double T) const;

    double compute_cv_ph(double p, double h) const;

    //! sigma(p,T)
    double compute_sigma_pT(double p, double T) const;

    double compute_sigma_ph(double p, double h) const;

    //! w(p,T)
    double compute_w_pT(double p, double T) const;

    double compute_w_ph(double p, double h) const;

    //! g(p,T)
    double compute_g_pT(double p, double T) const;

    double compute_g_ph(double p, double h) const;

    //! f(p,T)
    double compute_f_pT(double p, double T) const;

    double compute_f_ph(double p, double h) const;

    double compute_beta_pT(double p, double T) const;

    double compute_beta_ph(double p, double h) const;

    //! Gamma(p,h) and Gamma(p,T)
    double compute_gamma_ph(double p, double h) const;

    double compute_gamma_pT(double p, double T) const;

    //! Compute Prandtl(p,T)
    double compute_pr_pT(double p, double T) const;

    //General fluid properties
    double get_T_min() const;

    double get_T_max() const;

    double get_p_max() const;

    double get_p_min() const;

    double get_h_min() const;

    double get_h_max() const;

    double get_T_crit() const;

    double get_p_crit() const;

    double get_h_crit() const;

    /* Get molar mass in kg/mol */
    double get_mm() const;

    /* Partial derivatives (P,H) */
    double compute_d_T_d_p_h_ph(double p, double h) const;

    double compute_d_T_d_h_p_ph(double p, double h) const;

    /*
     * Not used since too "touchy"
    double compute_d_rho_d_p_h_ph(double p, double h) const;
    double compute_d_rho_d_h_p_ph(double p, double h) const;
    double compute_d_u_d_p_h_ph(double p, double h) const;
    double compute_d_u_d_h_p_ph(double p, double h) const;
    double compute_d_s_d_p_h_ph(double p, double h) const;
    double compute_d_s_d_h_p_ph(double p, double h) const;
    double compute_d_mu_d_p_h_ph(double p, double h) const;
    double compute_d_mu_d_h_p_ph(double p, double h) const;
    double compute_d_lambda_d_p_h_ph(double p, double h) const;
    double compute_d_lambda_d_h_p_ph(double p, double h) const;
    double compute_d_cp_d_p_h_ph(double p, double h) const;
    double compute_d_cp_d_h_p_ph(double p, double h) const;
    double compute_d_cv_d_p_h_ph(double p, double h) const;
    double compute_d_cv_d_h_p_ph(double p, double h) const;
    */

    /* Partial derivatives (P,T) */
    double compute_d_h_d_p_T_pT(double p, double T) const;

    double compute_d_h_d_T_p_pT(double p, double T) const;

    /*
     * Not used since too "touchy"
    double compute_d_rho_d_p_T_pT(double p, double T) const;
    double compute_d_rho_d_T_p_pT(double p, double T) const;
    double compute_d_u_d_p_T_pT(double p, double T) const;
    double compute_d_u_d_T_p_pT(double p, double T) const;
    double compute_d_s_d_p_T_pT(double p, double T) const;
    double compute_d_s_d_T_p_pT(double p, double T) const;
    double compute_d_mu_d_p_T_pT(double p, double T) const;
    double compute_d_mu_d_T_p_pT(double p, double T) const;
    double compute_d_lambda_d_p_T_pT(double p, double T) const;
    double compute_d_lambda_d_T_p_pT(double p, double T) const;
    double compute_d_cp_d_p_T_pT(double p, double T) const;
    double compute_d_cp_d_T_p_pT(double p, double T) const;
    double compute_d_cv_d_p_T_pT(double p, double T) const;
    double compute_d_cv_d_T_p_pT(double p, double T) const;
    */

    //
    //  Other methods
    //

    /*
    //! see Language
    std::ostream& print_On (std::ostream& stream=cout) const;
    //! see Language
    std::istream& read_On (std::istream& stream=cin);
    //! see Language
    const std::type_info& get_Type_Info () const;

    //Error metod
    void describe_error(const EOS_Internal_Error error, std::string & description) const;
    */
    double compute_gibbs(double T, double P) const;

protected:
    /*
    double sat_quality_;
    //! Handle phase id : phase_unkown, phase_gas or phase_liquid
    std::string handle_phase_;
    //! Backend
    std::string backend_ ;

    std::string fluid_name_; //!< Name of fluid
    std::string fluid_name_phase_ ; //!< Name of fluid with phase
    */
    static Region determine_region(double T, double P);

private:
    /*
    double molar_mass_; // Molar mass (kg/mol)
    static int type_Id;
    */

    std::shared_ptr<torch::jit::script::Module> model;
};

#endif //__NNPWS_HH__
