from dataclasses import dataclass, field

import chemicals as chem
from numpy import var
import pandas as pd
import math


def calc_mol_weight(composition: dict[str, float]) -> float:

    MW = 0

    for component in composition.keys():
        MW += composition[component] * chem.search_chemical(component).MW

    return MW


def ideal_gas_solver(P: float = None, T: float = None, V_bar: float = None) -> float:
    """
    Determine the missing parameter var_to_solve of the ideal gas equation from the other 2 given values

    T: float, temperature in K
    P: float, pressure in Pa
    V_bar: float, specific volume in m³/mol

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


def antoine_equation(component: str, P_vap: float = None, T: float = None) -> float:
    """
    Determine the vapour pressure or temperature from the antoine equation using base 10 for the exponential and logarithm
    P_vap: float, vapour pressure in Pa
    T: float, temperature in K
    """
    cas = chem.CAS_from_any(component)

    ant_data = chem.vapor_pressure.Psat_data_AntoinePoling.loc[cas]
    A = ant_data["A"]
    B = ant_data["B"]
    C = ant_data["C"]

    if P_vap == None:
        if T < ant_data["Tmin"] or T > ant_data["Tmax"]:
            print(
                f"Warning: out of T correlation range for component {component} {ant_data['Tmin']} < T < {ant_data['Tmax']}"
            )
        return 10 ** (A - B / (T + C))
    elif T == None:
        T = -(C + B / (math.log(P_vap, 10) - A))
        if T < ant_data["Tmin"] or T > ant_data["Tmax"]:
            print(
                f"Warning: out of T correlation range for component {component} {ant_data['Tmin']} < T < {ant_data['Tmax']}"
            )
        return T
    else:
        raise Exception("You didnt specifiy T or P_vap")


def gen_K_values(P: float, T: float, components: list[str]) -> dict[str, float]:
    """Determines the K_values of the components at given pressure and temperature, only Raoult's law at the moment"""
    result = {}
    K_value = None
    for component in components:
        P_vap = antoine_equation(T=T)
        result[component] = P_vap / P
    return result


def rachford_function(zs: list[float], ks: list[float], V_F: float) -> float:
    sum = 0
    for i, z in enumerate(zs):
        sum += (z * (ks[i] - 1)) / (1 + V_F * (ks[i] - 1))
    return sum


def rachford_funtion_first_derivative(zs: list[float], ks: list[float], V_F: float):
    sum = 0
    for i, z in enumerate(zs):
        sum += (-z * (ks[i] - 1) ** 2) / (1 + V_F * (ks[i] - 1) ** 2)
    return sum


def rachford_rice_flash(
    zs: list[float], ks: list[float]
) -> tuple([float, list[float], list[float], int]):
    """
    2 phase isothermal rachford_rice, using newtons method for now
    zs: list[float], overall mole fractions
    ks: list[float], k values of components\n
    returns:
    float: V_frac
    list:[float] xs liquid mole fractions
    list: [float] ys vapor mole fractions
    int: solve status: 0 = solved, 1 = iterations exceeded
    """

    # print(chem.rachford_rice.Rachford_Rice_solution(zs, ks))

    kmax = max(ks)
    kmin = min(ks)

    # guess for V/F
    V_frac = (
        ((kmax - kmin) * zs[ks.index(kmax)] - (1 - kmin)) / ((1 - kmin) * (kmax - 1))
        + (1 / (1 - kmin))
    ) / 2

    if V_frac > 1:
        guess = 1

    solved = False
    iterations = 0
    while solved == False:
        iterations += 1
        x = rachford_function(zs, ks, V_frac)
        x_prime = rachford_funtion_first_derivative(zs, ks, V_frac)
        new_V_frac = V_frac - x / x_prime
        V_frac = new_V_frac
        if iterations == 10000:
            status = 1
            break
        if math.fabs(x) < 0.0000000001:
            solved = True
            status = 0

    xs = []
    ys = []

    for i, z in enumerate(zs):
        xs.append(z / (1 + (ks[i] - 1) * V_frac))
        ys.append(ks[i] * z / (1 + (ks[i] - 1) * V_frac))
    print(V_frac, xs, ys, status)


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


@dataclass
class Stream:
    """Stream class for process flow streams"""

    id: str

    # Specified properties  2 required (currently requires T and P as specifications)
    T: float = None  # temperature in K
    P: float = None  # pressure in Pa
    V_frac: float = None  # vapour fraction

    # Flow and composition properties requires either overall_flows or total_flow_rate and overall_composition
    total_flow_rate: float = None  # flowrate mol/s
    overall_composition: dict[str, float] = None
    overall_flows: dict[str, float] = None  # mol/s

    stream_properties: dict[str, any] = None
    warnings: list[str] = field(init=False)

    def __post_init__(self):

        self.check_stream_definition()
        self.generate_stream_properties()
        self.warnings = []

        # Check if more than 1 component
        if len(self.overall_composition) == 1:
            if self.V_frac == None:
                ((component, v),) = self.overall_composition.items()
                P_vap = antoine_equation(component, T=self.T)
                if P_vap <= self.P:
                    # liquid phase assumed when P_vap = P for now as well
                    self.V_frac = 0
                elif P_vap > self.P:
                    # vapour phase
                    self.V_frac = 1
        else:
            if self.V_frac == None:

                pass
                # for component in self.overall_composition():

                # if the minimum vapour pressure for the components is above Tb
                # if antoine_equation("P",component,P=self.P):

                # self.multi_component_flash()

    def check_stream_definition(self):
        """Checks if the stream as defined is valid"""

        # fill out the overall flows , composition and total flow rate
        if self.overall_flows == None and (
            self.overall_composition == None or self.total_flow_rate == None
        ):
            raise Exception(
                "Not enough information, either provide overall_flows or both of overall_composition and total_flow_rate"
            )

        if self.overall_flows != None:

            if self.total_flow_rate == None:
                self.total_flow_rate = sum(self.overall_flows.values())

            if self.overall_composition == None:
                self.overall_composition = {}
                for component in self.overall_flows.keys():
                    print(component)
                    self.overall_composition[component] = (
                        self.overall_flows[component] / self.total_flow_rate
                    )
        else:
            self.overall_flows = {}
            for component in self.overall_composition.keys():
                self.overall_flows[component] = (
                    self.total_flow_rate * self.overall_composition[component]
                )

        for component in self.overall_composition.keys():
            try:
                chem.CAS_from_any(component)
            except ValueError:
                raise Exception(f"{component} is not a valid component, try again")

        if (
            self.T == None
            and (self.P == None or self.V_frac == None)
            or (self.P == None and self.V_frac == None)
        ):
            raise Exception(
                "Not enough information, provide two of temperature, pressure and Vapour fraction"
            )

        if self.V_frac is not None:
            if self.V_frac < 0.0 or self.V_frac > 1.0:
                raise Exception("Vapour_fraction must be between 0 and 1")

    def generate_stream_properties(self):
        self.stream_properties = {}
        self.stream_properties["MW"] = calc_mol_weight(self.overall_composition)
        # K value model selection needs to be refined
        self.stream_properties["V_L_Ks"] = {}

        # if self.V_frac == None:
        #     gen_K_values()

    # def multi_component_flash(self):

    #     ks = []
    #     zs = []
    #     for component in self.overall_composition.keys():
    #         pvap = antoine_equation("P", component, T=self.T)
    #         ks.append()
    #         zs.append(self.overall_composition[component])

    #     rachford_rice_flash(zs, ks)


flowsheet_1 = Flowsheet("flowsheet_1")
stream_1 = Stream(
    "stream_1",
    T=373,
    P=101325,
    total_flow_rate=1.0,
    overall_composition={"water": 1.0},
)

stream_2 = Stream(
    "stream_2", T=373.15, P=101325, overall_flows={"Water": 0.4, "ethanol": 0.6}
)

flowsheet_1.add_stream(stream_1)
flowsheet_1.add_stream(stream_2)
print(flowsheet_1.streams_list["stream_1"])
print(flowsheet_1.streams_list["stream_2"])

# rachford_rice_flash([0.5, 0.3, 0.2], [1.685, 0.742, 0.532])
