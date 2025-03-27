"""
进行三部分的数据准备
1. 三次插条等采样
2. 平滑滤波
3. 基线调整
"""
import numpy as np
import pywt
from scipy.interpolate import CubicSpline
from scipy.ndimage import grey_opening, grey_closing
from scipy.signal import savgol_filter
from getdata import get_all_patients_datas, get_datas

import matplotlib.pyplot as plt

def plot_patient_data(patient_datas, datas_labels,get_all=False, num=0,xlim=None):
    """
    绘制单个患者的折线图
    :param patient_data: 单个患者的数据字典，包含 "name" 和 "datas"
    """

    if(xlim==None):
        xlim=(0,1000)
    plt.figure(figsize=(10, 6))

    labels=datas_labels
    name = patient_datas[0]["name"]

    if(get_all):
        j=0
        for patient_data in patient_datas:
            for i, data in enumerate(patient_data["datas"]):
                x, y = data  # 提取 x 和 y
                # Filter data points within xlim range
                mask = (x >= xlim[0]) & (x <= xlim[1])
                x = x[mask]
                y = y[mask]
                plt.plot(x, y, label=labels[j], linewidth=1)  # 绘制折线图
                j=j+1
    else:
        j=0
        for patient_data in patient_datas:
            data = patient_data["datas"][num-1]
            x, y = data
            mask = (x >= xlim[0]) & (x <= xlim[1])
            x = x[mask]
            y = y[mask]
            plt.plot(x, y, label=labels[j], linewidth=1)
            j+=1

    plt.xlabel("X轴")
    plt.ylabel("Y轴")
    plt.title(f"{name} 折线图")
    plt.legend()
    plt.grid(True)
    plt.show()


def cubic_spline_resample(patient):
    resampled_datas=[]
    datas = patient["datas"]
    for data in datas:
        x, y = data
        cs = CubicSpline(x, y)
        num_samples = len(x)
        x_new=np.linspace(x[0],x[-1],num_samples)
        y_new = cs(x_new)
        data = np.array([x_new, y_new])
        resampled_datas.append(data)
    resampled_patient={
        "name": patient["name"],
        "datas": resampled_datas
    }
    return resampled_patient

def wavelet_smooth(patient):
    """
    使用 db4 小波对患者数据进行 4 层分解并平滑
    :param patient: 包含 "name" 和 "datas" 的患者数据字典
    :return: 平滑后的患者数据字典
    """
    smoothed_datas = []
    for data in patient["datas"]:
        x, y = data  # 提取 x 和 y 数据
        # 小波分解
        coeffs = pywt.wavedec(y, 'db4', level=4)
        # 对高频系数进行软阈值处理（平滑）
        threshold = np.sqrt( np.log(len(y)))  # 阈值可以根据需求调整
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        # 小波重构
        y_smooth = pywt.waverec(coeffs, 'db4')
        # 确保重构后的数据长度与原始数据一致
        y_smooth = y_smooth[:len(y)]
        # 将平滑后的数据存储
        smoothed_datas.append(np.array([x, y_smooth]))
    # 返回平滑后的患者数据
    smoothed_patient = {
        "name": patient["name"],
        "datas": smoothed_datas
    }
    return smoothed_patient

def baseline_correct(patient):
    """
    对患者数据进行基线调整
    :param patient: 包含 "name" 和 "datas" 的患者数据字典
    :return: 基线调整后的患者数据字典
    """
    corrected_datas = []
    for data in patient["datas"]:
        x, y = data
        # 创建一个足够大的结构元素进行开运算
        kernel_size = len(y) // 200  # 可调整大小
        # 开运算（先腐蚀后膨胀）
        baseline = grey_opening(y, size=kernel_size)
        baseline = grey_closing(baseline, size=kernel_size)
        baseline = savgol_filter(baseline, window_length=kernel_size, polyorder=3)  # 平滑基线
        # 减去基线
        y_corrected = y - baseline
        corrected_datas.append(np.array([x, y_corrected]))

    corrected_patient = {
        "name": patient["name"],
        "datas": corrected_datas
    }
    return corrected_patient

def peak_extract(patient):
    extracted_datas = []
    for data in patient["datas"]:
        x, y = data
        # 二阶连续小波变换
        scales = np.arange(1, 20)  # 尺度范围可调
        min_snr = 3
        coefficients = np.zeros((len(scales), len(y)))
        for i, scale in enumerate(scales):
            wavelet = pywt.cwt(y, scales=[scale], wavelet='mexh')[0][0]
            coefficients[i] = wavelet
            snr = np.abs(coefficients[i]).max() / np.std(coefficients[i])
            if(snr < min_snr):
                coefficients[i]=np.zeros(len(y))

        final_coefficients = np.abs(np.sum(coefficients, axis=0))
        extracted_datas.append(np.array([x, final_coefficients]))

    extracted_patient = {
        "name": patient["name"],
        "datas": extracted_datas
    }
    return extracted_patient

def prepare():
    all_patients_datas = get_all_patients_datas()
    prepared_datas,resampled_datas,smoothed_datas,corrected_datas,extracted_datas=[],[],[],[],[]
    for patient in all_patients_datas:
        resampled_data = cubic_spline_resample(patient)
        resampled_datas.append(resampled_data)
        smoothed_data = wavelet_smooth(resampled_data)
        smoothed_datas.append(smoothed_data)
        corrected_data = baseline_correct(smoothed_data)
        corrected_datas.append(corrected_data)
        extracted_data = peak_extract(corrected_data)
        extracted_datas.append(extracted_data)
        prepared_datas.append(extracted_data)

    return prepared_datas,resampled_datas,smoothed_datas,corrected_datas,extracted_datas


if __name__ == "__main__":
    (ad,rd,sd,cd,ed)=prepare()
    plot_patient_data([ad[0]], ["Origin"],get_all=False, num=1)
    plot_patient_data([rd[0]], ["Resampled"],get_all=False, num=1)
    plot_patient_data([sd[0]], ["Resampled"],get_all=False, num=1)
    plot_patient_data([cd[0]], ["Corrected"],get_all=False, num=1)
    
  