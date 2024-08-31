import os
import os
import shutil
import random


def change_label_class_ids(folder_path, new_class_id):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            modified_lines = []
            for line in lines:
                components = line.strip().split()
                components[0] = str(new_class_id)
                modified_line = ' '.join(components)
                modified_lines.append(modified_line+"\n")

            with open(file_path, 'w') as file:
                file.writelines(modified_lines)

# Example usage

#change_label_class_ids('C:\\work\\yolo\\Dataset\\labels\\val', 0)


def delete_unassociated_labels(label_folder, image_folder):
    label_files = os.listdir(label_folder)
    for label_file in label_files:
        if label_file.endswith('.txt'):
            image_file = label_file.replace('.txt', '.jpg')
            if not os.path.isfile(os.path.join(image_folder, image_file)):
                print(f"Deleted: {os.path.join(label_folder, label_file)}")
                os.remove(os.path.join(label_folder, label_file))




def delete_images_without_labels(images_folder, labels_folder):
    # 获取所有jpg文件的基础名称
    images = set([f[:-4] for f in os.listdir(images_folder) if f.endswith('.jpg')])

    # 获取所有txt文件的基础名称
    labels = set([f[:-4] for f in os.listdir(labels_folder) if f.endswith('.txt')])

    # 找出没有对应标签的图片
    images_without_labels = images - labels

    # 删除没有对应标签的图片
    for image_name in images_without_labels:
        file_path = os.path.join(images_folder, image_name + '.jpg')
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")



def move_files(source_folder, percentage=0.15):
    # 获取源文件夹中的所有文件
    valid_folder = 'D:\\AndroidProject\\det_garbage_yolo\\new\\valid\images'
    valid_labels = 'D:\\AndroidProject\\det_garbage_yolo\\new\\valid\labels'
    train_folder = 'D:\\AndroidProject\\det_garbage_yolo\\new\\train\images'
    train_labels = 'D:\\AndroidProject\\det_garbage_yolo\\new\\train\labels'
    save_images = 'D:\\AndroidProject\\det_garbage_yolo\\xiaoqu\\images'
    save_labels = 'D:\\AndroidProject\\det_garbage_yolo\\xiaoqu\\labels'
    all_files = [f[:-4] for f in os.listdir(source_folder) if f.endswith('.jpg')]
    for file_name in all_files:
        img_file = os.path.join(source_folder, file_name + '.jpg')
        label_file = os.path.join(source_folder, file_name + '.txt')
        shutil.copy(img_file, save_images)
        shutil.copy(label_file, save_labels)

    # 计算要移动的文件数量
    num_files_to_move = int(len(all_files) * percentage)
    # 随机选择要移动的文件
    valid_files = random.sample(all_files, num_files_to_move)
    train_files = set(all_files) - set(valid_files)


    # 移动文件
    for file_name in valid_files:
        img_file = os.path.join(source_folder, file_name + '.jpg')
        img_dst_file = os.path.join(valid_folder, file_name + '.jpg')
        label_file = os.path.join(source_folder, file_name + '.txt')
        label_dst_file = os.path.join(valid_labels, file_name + '.txt')
        shutil.move(img_file, img_dst_file)
        shutil.move(label_file, label_dst_file)
        print(f"Moved valid: {file_name}")

    for file_name in train_files:
        img_file = os.path.join(source_folder, file_name + '.jpg')
        img_dst_file = os.path.join(train_folder, file_name + '.jpg')
        label_file = os.path.join(source_folder, file_name + '.txt')
        label_dst_file = os.path.join(train_labels, file_name + '.txt')
        shutil.move(img_file, img_dst_file)
        shutil.move(label_file, label_dst_file)
        print(f"Moved train: {file_name}")








def merge_person_obj(label_folder,predict_folder):
    for file_name in os.listdir(predict_folder):
        if file_name.endswith(".txt"):
            pre_file_path = os.path.join(predict_folder, file_name)
            label_file_path = os.path.join(label_folder, file_name)
            with open(pre_file_path, 'r') as file:
               pre_lines = file.readlines()
            try:
                with open(label_file_path, 'r') as file:
                    label_lines = file.readlines()
            except:
                print(label_file_path)
                label_lines = []
            new_lines = []
            for line in pre_lines:
                new_lines.append(line)
            for line in label_lines:
                new_lines.append(line)
            with open(label_file_path, 'w') as file:
                file.writelines(new_lines)



def update_label():
    predict_folder = "C:\\Users\\97409\\Desktop\\MP4\\labels"
    images_folder = 'C:\\Users\\97409\\Desktop\\MP4\\frames'

    merge_person_obj(images_folder,predict_folder)
    # 使用函数
    delete_images_without_labels(images_folder, images_folder)
    delete_unassociated_labels(images_folder, images_folder)
    move_files(images_folder)



update_label()