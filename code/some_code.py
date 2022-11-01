from dataclasses import dataclass, field

import chemicals as chem
import numpy as np
import pandas as pd
import math


def calc_mol_weight(composition: dict[str, float]) -> float:
    """Calculates molecular weight of a mixed stream using the composition as a dictionary"""

    MW = 0

    for component in composition.keys():
        MW += composition[component] * chem.search_chemical(component).MW

    return MW


def ideal_gas_solver(P: float = None, T: float = None, V_bar: float = None) -> float:
    """
    Calculates the value of the missing parameter of the ideal gas equation from the other 2 given values\n
    T, temperature in K\n
    P, pressure in Pa\n
    V_bar, specific volume in m³/mol
    """
    R = 8.314  # J/mol.K
    if P == None:
        if T == None or V_bar == None:
            raise Exception("Not enough information, T and v_bar are required")
        return R * T / V_bar
    elif T == None:
        if P == None or V_bar == None:
            raise Exception("Not enough information, P and v_bar are required")
        return P * V_bar / R
    elif V_bar == None:
        if P == None or T == None:
            raise Exception("Not enough information, T and P are required")
        return R * T / P
    else:
        raise Exception("you didnt specify P, T or V_bar, provide 2")


def antoine_equation(
    comp: str, P_vap: float = None, T: float = None, warnings=[]
) -> float:
    """
    Determine the vapour pressure or temperature from the antoine equation using base 10 for the exponential and logarithm\n
    comp, component
    P_vap, vapour pressure in Pa\n
    T, temperature in K
    """
    cas = chem.CAS_from_any(comp)

    ant_data = chem.vapor_pressure.Psat_data_AntoinePoling.loc[cas]
    A = ant_data["A"]
    B = ant_data["B"]
    C = ant_data["C"]

    if P_vap == None:
        if T < ant_data["Tmin"] or T > ant_data["Tmax"]:
            warnings.append(
                f"Warning: out of T correlation range for component {comp} {ant_data['Tmin']} < T < {ant_data['Tmax']}"
            )
        return 10 ** (A - B / (T + C))
    elif T == None:
        T = -(C + B / (math.log(P_vap, 10) - A))
        if T < ant_data["Tmin"] or T > ant_data["Tmax"]:
            warnings.append(
                f"Warning: out of T correlation range for component {comp} {ant_data['Tmin']} < T < {ant_data['Tmax']}"
            )
        return T
    else:
        raise Exception("You didnt specifiy T or P_vap")


def gen_K_values(
    P: float, T: float, comps: dict[str, float], warnings=[]
) -> dict[str, float]:
    """
    Determines the K_values of the components at given pressure and temperature, only Raoult's law at the moment\n
    P, total pressure in Pa\n
    T, temperature in K
    """

    result = {}
    K_value = None
    for component in comps.keys():
        P_vap = antoine_equation(comp=component, T=T, warnings=warnings)
        result[component] = P_vap / P
    return result


def RR_equation(zs: list[float], Ks: list[float], V_frac: float) -> float:
    """
    Evaluates the Rachford Rice equation
    zs, overall mole fractions
    Ks, vapour liquid equilibrium constants
    V_Frac, vapour fraction
    """
    sum = 0
    for i, z in enumerate(zs):
        sum += (z * (Ks[i] - 1)) / (1 + V_frac * (Ks[i] - 1))
    return sum


def RR_equation_first_derivative(zs: list[float], Ks: list[float], V_frac: float):
    """
    Evaluates the first derivative of the Rachford Rice equation
    zs, overall mole fractions
    Ks, vapour liquid equilibrium constants
    V_Frac, vapour fraction
    """
    sum = 0
    for i, z in enumerate(zs):
        sum += (-z * (Ks[i] - 1) ** 2) / (1 + V_frac * (Ks[i] - 1) ** 2)
    return sum


def RR_flash(
    zs: dict[str, float], Ks: dict[str, float], warnings=[]
) -> tuple([float, dict[str, float], dict[str, float], int]):
    """
    2 phase isothermal rachford_rice, using newtons method for now
    zs, overall mole fractions
    Ks, vapour liquid equilibrium constants\n
    returns:
    V_frac: vapour fraction
    xs, liquid mole fractions
    ys, vapor mole fractions
    solve status, 0 = solved, 1 = iterations exceeded
    """
    comps = zs.keys()
    zs_list = []
    Ks_list = []
    for comp in comps:
        zs_list.append(zs[comp])
        Ks_list.append(Ks[comp])
    # print(chem.rachford_rice.Rachford_Rice_solution(zs_list, Ks_list))
    Kmax = max(Ks_list)
    Kmin = min(Ks_list)

    # guess for V/F
    V_frac = (
        ((Kmax - Kmin) * zs_list[Ks_list.index(Kmax)] - (1 - Kmin))
        / ((1 - Kmin) * (Kmax - 1))
        + (1 / (1 - Kmin))
    ) / 2

    if V_frac > 1:
        V_frac = 1

    solved = False
    iterations = 0
    while solved == False:
        iterations += 1
        x = RR_equation(zs_list, Ks_list, V_frac)
        x_prime = RR_equation_first_derivative(zs_list, Ks_list, V_frac)
        new_V_frac = V_frac - x / x_prime
        V_frac = new_V_frac
        if iterations == 10000:
            status = 1
            break
        if math.fabs(x) < 0.0000000001:
            solved = True
            status = 0

    xs = {}
    ys = {}

    for comp in comps:
        xs[comp] = zs[comp] / (1 + (Ks[comp] - 1) * V_frac)
        ys[comp] = Ks[comp] * zs[comp] / (1 + (Ks[comp] - 1) * V_frac)

    if V_frac < 0:
        V_frac = 0
        warnings.append(
            "V_frac calculated is less than 0, set to 0 , check calculation inputs"
        )
    elif V_frac > 1:
        V_frac = 1
        warnings.append(
            "V_frac calculated is greater than 1, set to 1 , check calculation inputs"
        )

    return (V_frac, xs, ys, status)


def check_stream_definition(stream):
    """Checks if the stream as defined is valid"""

    # make sure that enough data exists about the stream flows and composition
    if stream.overall_flows == None and (
        stream.overall_composition == None or stream.total_flow_rate == None
    ):
        raise Exception(
            "Not enough information, either provide overall_flows or both of overall_composition and total_flow_rate"
        )

    # In the case where the total flow rate is missing
    # but the overall flows are available
    if stream.total_flow_rate == None:
        if stream.overall_flows != None:
            stream.total_flow_rate = sum(stream.overall_flows.values())

    # Case where total_flow_rate and/or overall_flows were given but not the overall composition
    if stream.overall_composition == None:
        stream.overall_composition = {}
        if sum(stream.overall_flows.values()) != stream.total_flow_rate:
            raise Exception("overall_flows must sum to total_flow_rate")
        for component in stream.overall_flows.keys():
            stream.overall_composition[component] = (
                stream.overall_flows[component] / stream.total_flow_rate
            )
    # Case where total_flow_rate and overall composition were given but not overall_flows
    elif stream.overall_flows == None:
        stream.overall_flows = {}
        for component in stream.overall_composition.keys():
            stream.overall_flows[component] = (
                stream.total_flow_rate * stream.overall_composition[component]
            )

    # Check the composition sums to 1.0
    if sum(stream.overall_composition.values()) != 1.0:
        raise Exception("Overall composition must sum to 1")

    # Check the overall_flows match the overall composition
    for component in stream.overall_flows.keys():
        if (
            stream.overall_composition[component]
            != stream.overall_flows[component] / stream.total_flow_rate
        ):
            raise Exception(
                f"Composition does not match flow rate for component {component}"
            )

    # Check if all components have data in the chem module
    for component in stream.overall_composition.keys():
        try:
            chem.CAS_from_any(component)
        except ValueError:
            raise Exception(f"{component} is not a valid component, try again")

    if (
        stream.T == None
        and (stream.P == None or stream.V_frac == None)
        or (stream.P == None and stream.V_frac == None)
    ):
        raise Exception(
            "Not enough information, provide two of temperature, pressure and Vapour fraction"
        )

    if stream.V_frac is not None:
        if stream.V_frac < 0.0 or stream.V_frac > 1.0:
            raise Exception("Vapour_fraction must be between 0 and 1")


def generate_stream_properties(stream):
    stream.stream_properties = {}
    stream.stream_properties["MW"] = calc_mol_weight(stream.overall_composition)
    # K value model selection needs to be refined
    # So far this only solved if P and T are specified Vfrac and T and Vfrac and P solvers need to be implemented

    if stream.V_frac == None:
        stream.stream_properties["V_L_Ks"] = gen_K_values(
            P=stream.P,
            T=stream.T,
            comps=stream.overall_composition,
            warnings=stream.warnings,
        )
        # Check if more than 1 component
        if len(stream.overall_composition) == 1:
            if stream.V_frac == None:
                ((component, v),) = stream.overall_composition.items()
                P_vap = antoine_equation(
                    component, T=stream.T, warnings=stream.warnings
                )
                if P_vap <= stream.P:
                    # liquid phase assumed when P_vap = P for now as well
                    stream.V_frac = 0
                elif P_vap > stream.P:
                    # vapour phase
                    stream.V_frac = 1
        # multiple components
        else:
            if all(list(stream.stream_properties["V_L_Ks"].values())) < 1.0:
                stream.V_frac == 0
            elif all(list(stream.stream_properties["V_L_Ks"].values())) > 1.0:
                stream.V_frac == 1
            else:
                (
                    stream.V_frac,
                    stream.stream_properties["xs"],
                    stream.stream_properties["ys"],
                    Flash_status,
                ) = RR_flash(
                    stream.overall_composition,
                    stream.stream_properties["V_L_Ks"],
                    warnings=stream.warnings,
                )
                if Flash_status == 1:
                    stream.warnings.append("Rachford Rice maximum iterations exceeded")


class Flowsheet:
    """Flowsheet class"""

    def __init__(self, flowsheet_name):
        self.flowsheet_name: str = flowsheet_name

    streams_list = {}

    def add_stream(self, stream):
        if stream.id in self.streams_list.keys():
            raise Exception("Stream with this id already exists")
        else:
            self.streams_list[stream.id] = stream

    def solve(self):
        for stream in self.streams_list.values():
            if stream.is_inlet == True:
                check_stream_definition(stream)
                generate_stream_properties(stream)


@dataclass
class Stream:
    """Stream class for process flow streams"""

    id: str  # Stream name
    is_inlet: bool  # is the stream an inlet into the process

    # Specified properties  2 required (currently requires T and P as specifications)
    T: float = None  # temperature in K
    P: float = None  # pressure in Pa
    V_frac: float = None  # vapour fraction

    # Flow and composition properties requires either overall_flows or total_flow_rate and overall_composition
    total_flow_rate: float = None  # flowrate mol/s
    overall_composition: dict[str, float] = None
    overall_flows: dict[str, float] = None  # mol/s

    stream_properties: dict[str, any] = None
    warnings: list = field(default_factory=lambda: [])


flowsheet_1 = Flowsheet("flowsheet_1")
stream_1 = Stream(
    "stream_1",
    True,
    T=400,
    P=101325,
    # V_frac=2,
    total_flow_rate=5.0,
    overall_composition={"water": 0.4, "methanol": 0.6},
    overall_flows={"water": 2.0, "methanol": 3.0},
)

flowsheet_1.add_stream(stream_1)

flowsheet_1.solve()
print(flowsheet_1.streams_list["stream_1"])

# rachford_rice_flash([0.5, 0.3, 0.2], [1.685, 0.742, 0.532])
