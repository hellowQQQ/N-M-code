import cv2
import numpy as np
import skimage.filters as filters
from numpy.linalg import norm
import math
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def img_processing(I0):
    def x_func(img, x):
        return img[x, :], img[x, :]   
    
    def y_func(img, y):
        return img[:, y], img[:, y]
    
    def one_dimension_gauss(func, sigma):
        return cv2.GaussianBlur(func, (0, 0), sigma)    

    def gradient(vector, limit):
        abs_tatol_g = 0
        k_max = 5  # Line width setting
        before = 0  # Previous gray value
        graid_vector = np.zeros(len(vector))  # Create gradient vector
        bw = np.zeros(len(vector), dtype=np.uint8)  # Create texture vector

        # Calculate gradient vector
        for i in range(len(vector)):
            m = vector[i] - before  # Calculate the slope
            graid_vector[i] = m  # Store in gradient vector
            before = vector[i]  # Update previous point

        ads_graid_vector = np.abs(graid_vector)

        # Texture calculation
        for i in range(round(0.5 * (k_max - 1) + 1), len(graid_vector) - round(0.5 * (k_max - 1))):
            # Check slope change from negative to positive
            if graid_vector[i] >= 0 and graid_vector[i - 1] < 0:
                total_g = (ads_graid_vector[i - 2] + ads_graid_vector[i - 1] +
                            ads_graid_vector[i] + ads_graid_vector[i + 1] +
                            ads_graid_vector[i + 2])
                abs_tatol_g += total_g
                if total_g > 0.00:
                    bw[i - 2:i + 3] = 255

        bw = np.uint8(bw)
        return graid_vector, bw, bw
    

    roi_x0 = 30
    roi_y0 = 15
    roi_width = 500
    roi_high = 160
    
    # Define the region of interest (ROI)
    # roi_range = (roi_x0, roi_y0, roi_width, roi_high)
    ROI = I0[roi_y0:roi_y0 + roi_high, roi_x0:roi_x0 + roi_width]

    # Image processing
    lambda_value = 2.7961
    x_sigma = 3  # x-axis Gaussian blur parameter
    y_sigma = 3  # y-axis Gaussian blur parameter

    # Unsharp masking
    sp = filters.unsharp_mask(ROI, radius=10.0,
                                  amount=lambda_value, multichannel=False,               # 非銳化遮罩
                                  preserve_range=False)

    IMAGE = (255*sp).clip(0,255).astype(np.uint8)                                  # 圖像銳化

    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(4,4))                     # 自適應直方圖設定
    clahe_img = clahe.apply(IMAGE)


    # Gaussian filtering
    IMAGE = cv2.GaussianBlur(IMAGE, (5, 5), 0)

    nr, nc = IMAGE.shape

    x_bw_matrix = np.zeros((nr, nc), dtype=np.uint8)
    y_bw_matrix = np.zeros((nr, nc), dtype=np.uint8)

    # Y-axis processing
    for a in range(nc):  # Extracting y-axis gradient
        _, strength = y_func(IMAGE, a)
        strength_float64 = strength.astype(np.float64) / 255.0  # Normalizing
        strength = one_dimension_gauss(strength_float64, y_sigma)  # Curve blur
        _, bw, _ = gradient(strength, nc)  # Draw cross-section texture
        y_bw_matrix[:, a] = bw  # Accumulate y cross-section texture

    # X-axis processing
    for b in range(nr):  # Extracting x-axis gradient
        _, strength = x_func(IMAGE, b)
        strength_float64 = strength.astype(np.float64) / 255.0  # Normalizing
        strength = one_dimension_gauss(strength_float64, x_sigma)  # Curve blur
        _, bw, _ = gradient(strength, nr)  # Draw cross-section texture
        x_bw_matrix[b, :] = bw  # Accumulate x cross-section texture

    bw_matrix = cv2.add(x_bw_matrix, y_bw_matrix)  # Combine x and y textures

    # Morphological operations
    r_erode = 2
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_erode * 2 + 1, r_erode * 2 + 1))
    IMAGE = cv2.erode(bw_matrix, kernel_erode)

    r = 5
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
    IMAGE = cv2.dilate(IMAGE, kernel_dilate)
    cv2.imshow('pic',IMAGE)

    return IMAGE.astype(bool)  # Return logical image

def N_M(img,standard,C1,C2,C3,C4):

    def func(p,img,standard):
        x = p[0]
        y = p[1]
        angle = p[2]
        rows, cols = img.shape
        center = (cols // 2, rows // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, m, (cols, rows))

        # 平移圖像
        M = np.float32([[1, 0,  x], [0, 1,  y]])  # 平移矩陣
        img_S = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # 裁剪圖像
        img = img_S[20:140,20:480]
        # cv2.imshow('img',img)
        # cv2.waitKey(1)
        # 計算測試圖像
        img_or = cv2.bitwise_or(img, standard)
        img_or = img_or.astype(bool)
        img_and = cv2.bitwise_and(img, standard)
        img_and = img_and.astype(bool)

        J_or = np.sum(img_or)
        J_and = np.sum(img_and)
        # print(J_or)
        # print(J_and)

        J = -J_and / J_or
        
        # plt.pause(0.01)
        # cv2.imshow('img_standard',img_standard)
        # cv2.waitKey(1)
        return J,img_S,img
    
    
    
    new_size = (500, 160)
    new_image = np.full((new_size[1], new_size[0]), 255, dtype=np.uint8)
    x_offset = (new_size[0] - standard.shape[1]) // 2
    y_offset = (new_size[1] - standard.shape[0]) // 2
    new_image[y_offset:y_offset + standard.shape[0], x_offset:x_offset + standard.shape[1]] = standard
    # 創建 subplot
    fig, axs = plt.subplots(3, 1, figsize=(20, 10)) 


    axs[0].imshow(new_image, cmap='gray')
    axs[0].set_title('standard image', fontname='Times New Roman')
    axs[0].axis('off')  # 隱藏坐標軸
    plt.show(block=False)

    
    # standard = standard[20:140,20:480]
    kk = 0
    end_program  = 0
    max_kk = 10

    control_1 = C1
    control_2 = C2
    control_3 = C3
    control_4 = C4
    control = np.array([control_1,control_2,control_3,control_4])

    Kp = control[:,0]
    Ki = control[:,1]
    Kd = control[:,2]
    xx0 = np.column_stack((Kp, Ki, Kd))
    # print(xx0)
    p = xx0[0,:]
    n = len(p)
    #------計算單體初始性能指標----------
    xx = np.zeros_like(xx0)
    f = np.zeros(n+1)

    for i in range(n+1):
        f[i],img_s,img_target = func((xx0[i,:]),img,standard)

    while not end_program:
        # print('-----------------------第',kk,'次---------------------------')
        # print('排列前F = ', f)
        # print('排列前XX = ', xx)
        pr_a = 1
        #-----------大小排序------------
        f ,F_index = np.sort(f), np.argsort(f)
        # print('排列後F = ',f)
        # print('排列後F_index = ',F_index)
        for i in range(n+1):
            # print('xx0 = ',xx0)
            # print('xx = ',xx)
            # print('xx0[F_index[i],:] = ',xx0[F_index[i],:])
            # print('F_index[i] = ',F_index[i])
            xx[i,:] = xx0[F_index[i],:]
            
        # print('改變最佳效能參數')
        # print('xx = ',xx)
        j0 = f[0].copy()    #將目前最佳（小）的性能指標值存到 J0
        Jn = f[n].copy()    #將目前最差（大）的性能指標值存到 Jn
        Jn_1 = f[n].copy() #將目前次差（大）的性能指標值存到 Jn_1
        x0 = xx[0,:] # 將目前最佳的一組參數向量存到 X0
        Xn = xx[n,:].copy() #將目前最差的一組參數向量存到 Xn
        # print('x0,xn = ',x0,Xn)
        # print(Jn_1)
        x_ = np.mean(xx[:n, :], axis=0)
        # print(x_)
        # dX_ratio = norm(Xn - x0 / norm(x0))
        # dJ_ratio = abs( Jn-j0 )/abs(j0)
        if (kk>=max_kk):
            break
        Xnew = []
        Xr = x_ + x_ -Xn
        # print('X_ = ',x_)
        # print('Xn = ',Xn)
        # print('Xr = ',Xr)
        Jr,img_s,img_target = func(Xr,img,standard)
        # print('Jr J0 Jn_1 Jn = ',Jr,j0,Jn_1,Jn)
        if Jr < j0:  # Jr<J0 有以下兩種結果
            Xe = x_ + 2*(x_-Xn)
            Je,img_s,img_target = func(Xe,img,standard)
            # print('Xe,Je = ',Xe,Je)
            if Je < Jr:
                Xnew = Xe
                Jnew = Je
                # print('1')
            else:
                Xnew = Xr
                Jnew = Jr
                # print('2')
        elif Jr < Jn_1:
            Xnew = Xr
            Jnew = Jr
            # print('3')
        elif Jr < Jn:
            Xc = x_ + 0.5*(x_-Xn)
            Jc,img_s,img_target = func(Xc,img,standard)
            # print('Xc,Jc = ',Xc,Jc)
            if Jc <= Jn:
                Xnew = Xc
                Jnew = Jc
                # print('4')
        else:
            Xcc = x_ - 0.5*(x_-Xn)
            Jcc,img_s,img_target = func(Xcc,img,standard)
            # print('Xcc,Jcc = ',Xcc,Jcc)
            if Jcc <= Jn:
                Xnew = Xcc
                Jnew = Jcc
                # print('5')
        pri_J = Jnew
        if len(Xnew) == 0 :  # 檢查 Xnew 是否為空
            # print('執行多為收縮')
            for i in range(1, n + 1):  # 從第二個元素開始迭代（MATLAB 中索引從 1 開始）
                xx[i, :] = 0.5 * (xx[0, :] + xx[i, :])
                f[i],img_s, img_target = func(xx[i, :],img,standard)
            # print('XX = ',xx)
            # print('F = ',f)
            pri_J = f[0]
        else:
            # print('後續')
            xx[n, :] = Xnew
            f[n] = Jnew  # 很重要，不可漏掉
            # print('Xnew = ',Xnew)
            # print('Jnew = ',Jnew)
        xx0 = xx.copy()
        kk = kk+1
        # print(xx0)
        
        axs[1].imshow(img_s, cmap='gray')
        axs[1].set_title('comparison image', fontname='Times New Roman')
        axs[1].axis('off')  # 隱藏坐標軸
        axs[1].plot([20, 480, 480, 20, 20], [20, 20, 140, 140, 20], color='red', linewidth=2)


        axs[2].cla()  # 清空子圖
        new_image = np.full((new_size[1], new_size[0]), 255, dtype=np.uint8)
        new_image[y_offset:y_offset + img_target.shape[0], x_offset:x_offset + img_target.shape[1]] = img_target
        axs[2].imshow(new_image, cmap='gray')
        axs[2].set_title('comparison result', fontname='Times New Roman')
        axs[2].axis('off')  # 隱藏坐標軸
        calculation_result = "comparison result : "+ str(round(-pri_J * 100,1)) + " %"
        axs[2].text(0.5, -0.1, calculation_result, transform=axs[2].transAxes, 
                ha="center", fontsize=12, color='blue', fontname='Times New Roman')
        plt.draw()
        plt.pause(0.01)

    return j0,img_target



def main():
    # 建立Tkinter窗口並隱藏它
    Tk().withdraw()
    # 打開文件選擇對話框，讓使用者選擇圖像
    file_path = askopenfilename(title="選擇標準圖像", filetypes=[("Image ", "*.png;*.jpg;*.jpeg")])
    # 讀取選擇的圖像
    if file_path:
        img_standard = cv2.imread(file_path, 0)
        # 顯示選擇的圖像

    else:
        print("未選擇任何圖像")
    #standard = standard.astype(bool)                     # 轉成邏輯運算
    standard = img_standard[20:140,20:480]


    # 建立Tkinter窗口並隱藏它
    Tk().withdraw()
    # 打開文件選擇對話框，讓使用者選擇圖像
    file_path = askopenfilename(title="選擇比對圖像", filetypes=[("Image ", "*.png;*.jpg;*.jpeg")])
    # 讀取選擇的圖像
    if file_path:
        test = cv2.imread(file_path, 0)
        # 顯示選擇的圖像

    else:
        print("未選擇任何圖像")
    #test = test.astype(bool)
    #cv2.imshow('img',test)
    # 初始化變數
    sw_test = []
    sw_x = []
    sw_y = []
    sw_ag = []
    cc = 2
    pp = 0
    # 搜尋初始單體
    for x_s in range(-6, 7, cc):
        for y_s in range(-6, 7, cc):
            for ag in range(-3, 4, 3):  # 將 ag 設定為 0
                rows, cols = test.shape
                center = (cols // 2, rows // 2)
                m = cv2.getRotationMatrix2D(center, ag, 1.0)
                img = cv2.warpAffine(test, m, (cols, rows))

                # 平移圖像
                M = np.float32([[1, 0, 10 * x_s], [0, 1, 10 * y_s]])  # 平移矩陣
                img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

                # 裁剪圖像
                img = img[20:140,20:480]

                # 計算測試圖像
                img_or = cv2.bitwise_or(img, standard)
                img_or = img_or.astype(bool)
                img_and = cv2.bitwise_and(img, standard)
                img_and = img_and.astype(bool)

                J_or = np.sum(img_or)
                J_and = np.sum(img_and)
                # print(J_or)
                # print(J_and)

                J = -J_and / J_or
                
                
                # 存儲結果
                sw_test.append(J)
                sw_x.append(10 * x_s)
                sw_y.append(10 * y_s)
                sw_ag.append(ag)
                pp += 1
                
    sw_index = np.argsort(sw_test)  # 獲取排序的索引
    C1 = [sw_x[sw_index[0]] ,sw_y[sw_index[0]] ,sw_ag[sw_index[0]]]  
    C2 = [sw_x[sw_index[0]] + math.ceil(5 * cc) ,sw_y[sw_index[0]] ,  3 ] 
    C3 = [sw_x[sw_index[0]],sw_y[sw_index[0]] + math.ceil(5 * cc)  , -3 ] 
    C4 = [sw_x[sw_index[0]] + math.ceil(5 * cc),sw_y[sw_index[0]] + math.ceil(5 * cc), 0 ] 
    
    # print('-----------------------外--------------------------------')
    # print(sw_test[sw_index[0]])
    # print(C1)
    # print(C2)
    # print(C3)
    # print(C4)
    # print('-------------------------------------------------------')

    [J_final,img_final] = N_M(test,standard,C1,C2,C3,C4)
    # print(J_final)
    # cv2.imshow('img_standard',standard)
    # cv2.imshow('img_final',img_final)
    
    # rows, cols = test.shape
    # center = (cols // 2, rows // 2)
    # m = cv2.getRotationMatrix2D(center, 0, 1.0)
    # img = cv2.warpAffine(test, m, (cols, rows))

    # # 平移圖像
    # M = np.float32([[1, 0, 10 * -2], [0, 1, 10 * 0]])  # 平移矩陣
    # img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # # 裁剪圖像
    # img = img[20:140,20:480]

    # # 計算測試圖像
    # img_or = cv2.bitwise_or(img, standard)
    # img_or = img_or.astype(bool)
    # img_and = cv2.bitwise_and(img, standard)
    # img_and = img_and.astype(bool)

    # J_or = np.sum(img_or)
    # J_and = np.sum(img_and)
    # # print(J_or)
    # # print(J_and)

    # J = -J_and / J_or
    
    # print(J)
    
    # input("按下 Enter 鍵以繼續...")
    cv2.waitKey(0)
    cv2.destroyAllWindows
main()




