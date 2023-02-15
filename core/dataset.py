import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
from config import INPUT_SIZE
import PIL


class Cherry():
    def __init__(self, root, is_train=True, is_validation = False, data_len=None):
        self.root = root
        self.is_train = is_train
        self.is_validation = is_validation
        # img_txt_file = open(os.path.join(self.root, 'images.txt'))
        # label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        # train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        # img_name_list = []
        # for line in img_txt_file:
        #     img_name_list.append(line[:-1].split(' ')[-1])
        # label_list = []
        # for line in label_txt_file:
        #     label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        # train_test_list = []
        # for line in train_val_file:
        #     train_test_list.append(int(line[:-1].split(' ')[-1]))
        # train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        # test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        # if self.is_train:
        #     self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
        #                       train_file_list[:data_len]]
        #     self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        # if not self.is_train:
        #     self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
        #                      test_file_list[:data_len]]
        #     self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]


        # cherry88
        id_and_path = np.genfromtxt(os.path.join('./', 'cherry-dataset/cherry_id.txt'), dtype=str)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        validation_data = []
        validation_lables = []
        # print('Data preprocessing, storage files')
        # pbar = tqdm(total=len(id_and_path))
        num_load = 0
        for i in id_and_path:
            label = int(i[1])
            if len(str(i[0])) == 3:
                add_path = str(i[0])
            elif len(str(i[0])) == 2:
                add_path = '0' + str(i[0])
            else:
                add_path = '00' + str(i[0])
            filename_names = []
            path = os.path.join('./', 'cherry-dataset/cherry_jpg/')
            path = path + add_path
            for filename in os.listdir(path):
                filename_names.append(filename)
            data_class = []
            label_class = []
            # print("文件列表名称：",filename_names)
            for id in filename_names:
                image = PIL.Image.open(os.path.join(path, id))
                # label = int(id_and_path[id, 1][:3]) - 1

                # Converts gray scale to RGB
                if image.getbands()[0] == 'L':
                    image = image.convert('RGB')

                np_image = np.array(image)
                num_load += 1
                # print("加载第{0}张图片".format(num_load))

                image.close()

                data_class.append(np_image)
                label_class.append(label)

            # 数据集划分
            if self.is_train:
                for i in data_class[:int(0.6 * len(data_class) + 1)]:
                    train_data.append(i)
                    train_labels.append(label)

            if not self.is_train and not self.is_validation:
                for i in data_class[int(0.6 * len(data_class) + 1):int(0.9 * len(data_class))]:
                    test_data.append(i)
                    test_labels.append(label)

            if not self.is_train and self.is_validation:
                for i in data_class[int(0.9 * len(data_class)):]:
                    validation_data.append(i)
                    validation_lables.append(label)

            del data_class[:]
            del data_class

        if self.is_train:
            self.train_img = train_data
            self.train_label = train_labels
        if not self.is_train:
            self.test_img = test_data
            self.test_label = test_labels
        if self.is_validation:
            self.validation_img = validation_data
            self.validation_label = validation_lables


    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        elif not self.is_validation:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        else:
            img, target = self.validation_img[index], self.validation_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        elif not self.is_validation:
            return len(self.test_label)
        else:
            return len(self.validation_label)


if __name__ == '__main__':
    dataset = Cherry(root='./cherry-dataset')
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = Cherry(root='./cherry-dataset', is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
