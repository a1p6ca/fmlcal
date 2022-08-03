from time import perf_counter
import os
import sys
from itertools import product

from scipy.stats import binom
import pandas as pd
import numpy as np


def timeit(func):

    def wrapper(*args, **kwargs):
        from time import perf_counter
        t0 = perf_counter()
        f = func(*args, **kwargs)
        t1 = perf_counter()
        print(t1-t0)
        return f

    return wrapper


def binomial_simulation(m: int, p: float):
    """
    离子含m个原子p，其最丰富同位素离子中p的同位素（目前，最多两种）组成

    例如：Cl10 -> [35]Cl8[37]Cl2
    binomial_simulation(10, .76) -> (8, 2)
    """
    tpl = ((_, m - _) for _ in range(m + 1))
    return np.array(max(tpl, key=lambda x: binom.pmf(x[0], m, p)), dtype=np.int16)


def unpacking_with_self(func):

    def wrapper(self, *args):
        if not args:
            return func(self, args)

        if isinstance(args[0], (tuple, list)):
            return func(self, args[0])
        else:
            return func(self, args)

    return wrapper


def unpacking_without_self(func):
    def wrapper(*args):

        if isinstance(args[0], (tuple, list)):
            return func(args[0])
        else:
            return func(args)

    return wrapper


class ChargeFix:
    fixed_mass = {-1: np.float32(0.0005484),
                  0:  np.float32(0),
                  1:  np.float32(-0.0005484)}

    @classmethod
    def get_fixex_mass(cls, charge: int):
        assert charge in {-1, 0, 1}, 'multiple charges are not supported'
        return cls.fixed_mass[charge]


class IsoElement:
    """
    同位素类
    """

    def __init__(self, name: str, mass: float, abundant: float = 1.):
        self.name = name
        self.mass = mass
        self.abundant = abundant

    def __repr__(self):
        return self.name


class IsoElementLst:

    def __init__(self, iso_lst: list[IsoElement]):
        if iso_lst is not None:
            self.num = len(iso_lst)
            self.name = [_.name for _ in iso_lst]
            self.mass = np.array([_.mass for _ in iso_lst], dtype=np.float32)
            self.abundant = np.array([_.abundant for _ in iso_lst])
            self.abundant = self.abundant / sum(self.abundant)

    def __repr__(self):
        return ','.join(self.name)


class Element(IsoElement):
    """
    元素类
    """

    def __init__(self, name: str, mass: float, iso=None, abundant=None):
        super().__init__(name, mass, abundant)
        self.iso = IsoElementLst(iso)
        self.distribution = {}

    # 根据相对丰度，生成至多m个原子时的同位素分布
    def cal_iso_distribution(self):
        # 只有一种同位素的情况
        if self.iso.num == 1:
            self.distribution.update({_: {'comb': (x := np.array([_], dtype=np.int16)),
                                          'mass': np.dot(x, self.iso.mass)} for _ in range(0, 100)})
            return

        # 有两种同位素的情况
        for _ in range(0, 100):
            self.distribution.update({_: {'comb': (x := np.array(binomial_simulation(_, self.iso.abundant[0]))),
                                          'mass': np.dot(x, self.iso.mass)}})
        return

    # 查询n个元素时同位素分布
    def get_distribution(self, nmin: int, nmax: int):
        assert self.distribution, "该元素还未生成分布"
        return np.array([ele_Cl.distribution[_]['mass'] for _ in range(nmin, nmax)])

    def create_condition(self, nmin, nmax):
        return SingleElementCondition(self, nmin, nmax)

    def __repr__(self):
        return self.name


class SingleElementCondition:
    """
    单一元素及原子个数取值最大最小值
    """

    def __init__(self, element, nmin: int, nmax: int):
        assert 0 <= nmin <= nmax, '最大最小值错误'
        self.element = element
        self.nmin = nmin
        self.nmax = nmax


class FullElementCondition:
    """
    一组元素及原子个数取值最大最小值
    """

    def __init__(self):
        self.condition_dict = {}
        self.ele_comp = None
        self.mass_data = None

    @unpacking_with_self
    def update(self, new_condition: [SingleElementCondition] = []):
        self.condition_dict.update({_.element: {'name': _.element.name,
                                                'mass': _.element.mass,
                                                'nmin': _.nmin,
                                                'nmax': _.nmax}
                                    for _ in new_condition})
        self.ele_comp = FormulaElementComposition(self.condition_dict.keys())

    def generate_mass_data(self):
        self.mass_data = MassData(self.condition_dict)

    @staticmethod
    def _cal_endpoints(flag, threshold, theom_mass):
        if flag == 'ppm':
            return (1 - 1e-6 * threshold) * theom_mass, (1e-6 * threshold + 1) * theom_mass
        elif flag == 'mmu':
            return theom_mass - threshold / 1e3, theom_mass + threshold / 1e3
        else:
            return 0, 0

    def cal_formular(self, *mass_lst, charge: int = 1, flag='ppm', threshold=5):

        results = []
        fixed_mass = ChargeFix.get_fixex_mass(charge)

        for _ in mass_lst:
            endpoints = self._cal_endpoints(flag, threshold, _)
            hit = np.bitwise_and(self.mass_data.fml_mass + fixed_mass > endpoints[0],
                                 self.mass_data.fml_mass + fixed_mass < endpoints[1])

            result = [Formula(self.ele_comp, comp, mass, charge)
                      for comp, mass in zip((self.mass_data.ele_comp[hit]), self.mass_data.fml_mass[hit])]
            results.append(result)
        return results


class MassData:
    """
    根据元素约束生成表格（元素组成、离子质量）
    """

    def __init__(self, condition):
        self.condition = condition
        self.ele_comp = None
        self.ele_mass = None
        self.fml_mass = None
        self.mass = None
        self.initialization()

    def initialization(self):
        # 查询到的精确质量
        tmp = list(range((x := self.condition[_])['nmin'], x['nmax']+1) for _ in self.condition.keys())

        # 质量组成：各元素原子数的笛卡尔乘积
        self.ele_comp = np.array(list(product(*tmp)), dtype=np.int16)

        # 质量数组：各元素质量的笛卡尔乘积
        self.ele_mass = ([k.distribution[num]['mass'] for num in range(v['nmin'], v['nmax'] + 1)]
                         for k, v in self.condition.items())
        self.ele_mass = np.array(list(product(*self.ele_mass)))

        # 分子质量
        self.fml_mass = self.ele_mass.sum(axis=1)


class FormulaElementComposition:
    """
    分子式中，采用了哪些元素
    """

    def __init__(self, element: list[Element]):
        assert len(set(element)) == len(element), "有重复元素"
        self.element_lst = element
        self.element_name = self.get_element_name()

    def get_element_name(self):
        return [_.name for _ in self.element_lst]

    def __repr__(self):
        return ','.join(self.element_name)


class Formula:

    def __init__(self,
                 ele_comp: FormulaElementComposition,
                 ele_num: list[int],
                 acu_mass: np.float32 = None,
                 charge: int = 0
                 ):
        self.ele_comp = ele_comp
        self.ele_num = ele_num
        self.acu_mass = acu_mass
        self.fixed_mass = acu_mass + ChargeFix.get_fixex_mass(charge) if acu_mass else None
        self.formula = self.get_formula()

    def get_formula(self, skip0=True, skip1=True):
        f = ''
        for e, n in zip(self.ele_comp.element_name, self.ele_num):
            if n == 0 & skip0:
                continue
            if n == 1 & skip1:
                f += f'{e}'
            else:
                f += f'{e}{n}'
        return f

    def __repr__(self):
        return self.formula


ele_C12 = IsoElement('C', 12.00000)
ele_C = Element('C', 12, [ele_C12])

ele_H1 = IsoElement('H', 1.0078250319)
ele_H = Element('H', 1, [ele_H1])

ele_O12 = IsoElement('O', 15.994914619)
ele_O = Element('O', 16, [ele_O12])

ele_N14 = IsoElement('N', 14.003074004)
ele_N = Element('N', 14, [ele_N14])

ele_S32 = IsoElement('S', 31.97207117)
ele_S = Element('S', 32, [ele_S32])

ele_P31 = IsoElement('P', 30.973761998)
ele_P = Element('P', 31, [ele_P31])

ele_Cl35 = IsoElement('Cl35', 34.9688527, 100.)
ele_Cl37 = IsoElement('Cl37', 36.9659026, 31.96)
ele_Cl = Element('Cl', 35, [ele_Cl35, ele_Cl37])

ele_Br79 = IsoElement('[79Br]', 78.9183376, 100)
ele_Br81 = IsoElement('[81Br]', 80.9162910, 97.28)
ele_Br = Element('Br', 79, [ele_Br79, ele_Br81])

comp1 = FormulaElementComposition([ele_C, ele_Cl, ele_Br])
f1 = Formula(comp1, [6, 4, 2])

ele_C.cal_iso_distribution()
ele_H.cal_iso_distribution()
ele_O.cal_iso_distribution()
ele_N.cal_iso_distribution()
ele_S.cal_iso_distribution()
ele_Cl.cal_iso_distribution()
ele_Br.cal_iso_distribution()

C_condition = ele_C.create_condition(1, 30)
H_condition = ele_H.create_condition(0, 60)
O_condition = ele_O.create_condition(0, 5)
N_condition = ele_N.create_condition(0, 5)
S_condition = ele_S.create_condition(0, 5)
P_condition = ele_P.create_condition(0, 2)
Cl_condition = ele_Cl.create_condition(0, 10)
Br_condition = ele_Br.create_condition(0, 10)

condition1 = FullElementCondition()
condition1.update(C_condition,
                  H_condition,
                  O_condition,
                  N_condition,
                  S_condition,
                  #P_condition,
                  Cl_condition,
                  Br_condition)

time1 = perf_counter()
condition1.generate_mass_data()
time2 = perf_counter()
print(time2-time1)




if __name__ == '__main__':
    pass
    # TODO
    # workflow
    # 第一步：定义元素(同位素) -> 用数据库完成，保存在本地
    # 第二步：元素初始化同位素分布 -> 用数据库完成，保存在本地
    # 第三步：分子式元素组成


condition1.cal_formular = timeit(condition1.cal_formular)
mass_lst = np.random.randint(100, 500, 100)
trash = condition1.cal_formular(*mass_lst, flag='ppm', threshold=5)
