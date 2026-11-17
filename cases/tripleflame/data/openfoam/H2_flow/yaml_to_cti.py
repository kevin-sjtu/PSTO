#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 YAML 文件转换为 CTI 格式
修复了 duplicate=True 导致的 CanteraError
"""

import yaml
import re
import os
import sys

def format_atoms(composition):
    """将 YAML 的 composition 格式转换为 CTI 的 atoms 格式"""
    parts = []
    for element, count in composition.items():
        parts.append(f"{element}:{int(count)}")
    return " ".join(parts)

def format_nasa_coeffs(coeffs):
    """格式化 NASA 系数"""
    return f"[{coeffs[0]:.8g}, {coeffs[1]:.8g}, {coeffs[2]:.8g}, {coeffs[3]:.8g}, {coeffs[4]:.8g}, {coeffs[5]:.8g}, {coeffs[6]:.8g}]"

def format_transport(transport):
    """格式化 transport 参数"""
    params = []
    geom_map = {"atom": "atom", "linear": "linear", "nonlinear": "nonlinear"}
    params.append(f'geom="{geom_map[transport["geometry"]]}"')
    params.append(f'diam={transport["diameter"]}')
    params.append(f'well_depth={transport["well-depth"]}')
    
    if "polarizability" in transport:
        params.append(f'polar={transport["polarizability"]}')
    if "dipole" in transport:
        params.append(f'dipole={transport["dipole"]}')
    if "rotational-relaxation" in transport:
        params.append(f'rot_relax={transport["rotational-relaxation"]}')
    
    return ", ".join(params)

def format_efficiencies(efficiencies):
    """格式化 efficiencies"""
    parts = []
    for species, value in efficiencies.items():
        parts.append(f"{species}:{value}")
    return " ".join(parts)

def parse_pressure(pressure_value):
    """解析压力值，从字符串中提取数值并转换单位"""
    if isinstance(pressure_value, (int, float)):
        return float(pressure_value)
    
    # 如果是字符串，提取数值和单位
    if isinstance(pressure_value, str):
        # 匹配数值和单位，例如 "1 atm", "101325 Pa", "1.0 bar", "1.01325e+05 Pa" 等
        match = re.match(r'([0-9.eE+-]+)\s*([A-Za-z]+)', pressure_value.strip())
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            
            # 转换为 Pa (Cantera 默认使用 Pa)
            if unit == 'atm':
                return value * 101325.0
            elif unit == 'bar':
                return value * 100000.0
            elif unit == 'pa' or unit == 'pascal':
                return value
            elif unit == 'kpa':
                return value * 1000.0
            elif unit == 'mpa':
                return value * 1000000.0
            else:
                # 未知单位，直接返回数值（假设已经是 Pa）
                return value
        else:
            # 如果无法解析，尝试直接转换为浮点数
            try:
                return float(pressure_value)
            except ValueError:
                return 101325.0  # 默认值 1 atm
    
    return 101325.0  # 默认值 1 atm


def parse_float_with_unit(value, default=0.0):
    """解析类似 '1.033e+05 cal/mol' 这样的带单位字符串，返回数值部分"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # 提取第一个浮点或科学计数法数字
        match = re.search(r'([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)', value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        try:
            return float(value)
        except ValueError:
            return default
    return default

def convert_reaction(reaction):
    """转换反应格式"""
    equation = reaction["equation"]
    # 移除注释
    equation = re.sub(r'\s*#.*$', '', equation).strip()
    
    # 处理可逆反应
    if "<=>" in equation:
        equation = equation.replace("<=>", "=>")
    
    # 判断是否为 falloff 反应
    is_falloff = reaction.get("type") == "falloff"
    
    # 对于非 falloff 反应，移除 (+ M) 标记
    if not is_falloff:
        equation = equation.replace("(+ M)", "").replace("(+M)", "")
    
    # 清理多余的空格，但保护 (+ M)
    if is_falloff:
        equation = equation.replace("(+ M)", "(_PLUS_M_)")
        equation = equation.replace("(+M)", "(_PLUS_M_)")
    
    equation = re.sub(r'\s+', ' ', equation).strip()
    equation = re.sub(r'\s*=>\s*', ' => ', equation)
    equation = re.sub(r'\s*<=\s*', ' <= ', equation)
    equation = re.sub(r'\s+', ' ', equation).strip()
    
    if is_falloff:
        equation = equation.replace("(_PLUS_M_)", "(+ M)")

    # ------------------- 核心修改部分开始 -------------------
    # 检查 YAML 中是否有 duplicate 标记，并生成 options 字符串
    duplicate_str = ""
    if reaction.get("duplicate", False):
        duplicate_str = ', options="duplicate"'
    # ------------------- 核心修改部分结束 -------------------

    if reaction.get("type") == "three-body":
        rate_const = reaction["rate-constant"]
        A = rate_const["A"]
        b = rate_const.get("b", 0.0)
        Ea = parse_float_with_unit(rate_const.get("Ea", 0.0))
        
        eff_str = ""
        if "efficiencies" in reaction:
            eff_str = f',\n    efficiencies = "{format_efficiencies(reaction["efficiencies"])}"'
        
        # 在返回值中加入 duplicate_str
        return f'three_body_reaction("{equation}", [{A:.8g}, {b:.8g}, {Ea:.8g}]{eff_str}{duplicate_str})'
    
    elif reaction.get("type") == "falloff":
        rate_const_low = reaction["low-P-rate-constant"]
        rate_const_high = reaction["high-P-rate-constant"]
        troe = reaction.get("Troe", {})
        
        A_low = rate_const_low["A"]
        b_low = rate_const_low.get("b", 0.0)
        Ea_low = parse_float_with_unit(rate_const_low.get("Ea", 0.0))
        
        A_high = rate_const_high["A"]
        b_high = rate_const_high.get("b", 0.0)
        Ea_high = parse_float_with_unit(rate_const_high.get("Ea", 0.0))
        
        eff_str = ""
        if "efficiencies" in reaction:
            eff_str = f',\n    efficiencies = "{format_efficiencies(reaction["efficiencies"])}"'
        
        troe_str = ""
        if troe:
            troe_params = []
            if "A" in troe:
                troe_params.append(f'A={troe["A"]:.8g}')
            if "T3" in troe:
                troe_params.append(f'T3={troe["T3"]:.8g}')
            if "T1" in troe:
                troe_params.append(f'T1={troe["T1"]:.8g}')
            if "T2" in troe:
                troe_params.append(f'T2={troe["T2"]:.8g}')
            if troe_params:
                troe_str = f',\n    falloff = Troe({", ".join(troe_params)})'
        
        # 在返回值中加入 duplicate_str
        return f'''falloff_reaction("{equation}",
    kf0=[{A_low:.8g}, {b_low:.8g}, {Ea_low:.8g}],
    kf=[{A_high:.8g}, {b_high:.8g}, {Ea_high:.8g}]{troe_str}{eff_str}{duplicate_str})'''
    
    else:
        # 普通反应
        rate_const = reaction["rate-constant"]
        A = rate_const["A"]
        b = rate_const.get("b", 0.0)
        Ea = parse_float_with_unit(rate_const.get("Ea", 0.0))
        
        # 在返回值中加入 duplicate_str
        return f'reaction("{equation}",\n    [{A:.8g}, {b:.8g}, {Ea:.8g}]{duplicate_str})'

def main():
    # 读取 YAML 文件（支持命令行指定，默认使用 ES80_H2-7-16.yaml）
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
    else:
        yaml_path = "ES80_H2-7-16.yaml"

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 找不到 {yaml_path} 文件")
        return
    
    # 输出 CTI 文件名：与输入 YAML 同名但后缀为 .cti
    base, _ = os.path.splitext(yaml_path)
    cti_path = base + ".cti"

    # 开始构建 CTI 文件
    cti_lines = []
    
    # units
    units = data.get("units", {})
    length = units.get("length", "cm")
    time = units.get("time", "s")
    quantity = units.get("quantity", "mol")
    act_energy = units.get("activation-energy", "cal/mol")
    
    cti_lines.append(f'units(length="{length}", time="{time}", quantity="{quantity}", act_energy="{act_energy}")')
    cti_lines.append("")
    
    # gas phase
    if "phases" in data and len(data["phases"]) > 0:
        phase = data["phases"][0]
        species_list = " ".join(phase["species"])
        elements_str = " ".join(phase["elements"])
        state = phase["state"]
        
        # 解析温度和压力
        temperature_raw = state.get("T", 300.0)
        temperature = parse_float_with_unit(temperature_raw, 300.0)
        pressure_raw = state.get("P", "1 atm")
        pressure = parse_pressure(pressure_raw)
        
        cti_lines.append(f'gas = ideal_gas(')
        cti_lines.append(f'    name="gas",')
        cti_lines.append(f'    elements="{elements_str}",')
        cti_lines.append(f'    species="""')
        cti_lines.append(f'        {species_list}')
        cti_lines.append(f'    """,')
        cti_lines.append(f'    reactions="all",')
        cti_lines.append(f'    initial_state=state(temperature={temperature}, pressure={pressure})')
        cti_lines.append(f')')
        cti_lines.append("")
    
    # species
    for species in data.get("species", []):
        cti_lines.append(f'species(')
        cti_lines.append(f'    name="{species["name"]}",')
        atoms_str = format_atoms(species["composition"])
        cti_lines.append(f'    atoms="{atoms_str}",')
        
        # thermo
        if "thermo" in species:
            thermo = species["thermo"]
            temp_ranges = thermo["temperature-ranges"]
            data_list = thermo["data"]
            
            cti_lines.append(f'    thermo=(')
            for i, temp_range in enumerate(range(len(temp_ranges) - 1)):
                T_low = temp_ranges[i]
                T_high = temp_ranges[i + 1]
                coeffs = data_list[i]
                coeffs_str = format_nasa_coeffs(coeffs)
                cti_lines.append(f'        NASA([{T_low:.1f}, {T_high:.1f}],')
                cti_lines.append(f'             {coeffs_str}),')
            cti_lines.append(f'    ),')
        
        # transport
        if "transport" in species:
            transport = species["transport"]
            transport_str = format_transport(transport)
            cti_lines.append(f'    transport=gas_transport(')
            cti_lines.append(f'        {transport_str}')
            cti_lines.append(f'    )')
        
        cti_lines.append(f')')
        cti_lines.append("")
    
    # reactions
    for reaction in data.get("reactions", []):
        try:
            cti_lines.append(convert_reaction(reaction))
            cti_lines.append("")
        except Exception as e:
            print(f"转换反应时出错: {reaction.get('equation', 'unknown')}")
            raise e
    
    # 写入文件
    output = "\n".join(cti_lines)
    with open(cti_path, "w") as f:
        f.write(output)
    
    print(f"转换完成！已生成 {cti_path} 文件")

if __name__ == "__main__":
    main()