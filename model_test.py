import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import FashionMNIST
from model import ResNet, Residual
from tqdm import tqdm #
from PIL import Image # 导入 tqdm

def test_data_process():
    ROOT_TRAIN_PATH = 'data/train'
    normalize = transforms.Normalize([0.22890999, 0.1963964,  0.14335695], [0.09950233, 0.07996743, 0.06593084])
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    test_data = ImageFolder(root=ROOT_TRAIN_PATH, transform=test_transform)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1, # 通常测试时 batch_size 可以设置为 1
                                      shuffle=True, # 测试集通常不需要打乱
                                      num_workers=0)
    return test_dataloader # 修正：返回数据加载器

def test_model_process(model, test_dataloader):
    # 自动检测并使用GPU，如果可用的话
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    model = model.to(device) # 将模型移动到指定设备
    model.eval() # 将模型设置为评估模式，只调用一次

    test_corrects = 0.0
    test_num = 0
    with torch.no_grad(): # 在此上下文管理器中，禁用梯度计算，节省内存和计算
        # 使用 tqdm 包装 test_dataloader，显示进度条
        for test_data_x, test_data_y in tqdm(test_dataloader, desc="Testing"):
            test_data_x = test_data_x.to(device) # 将数据移动到指定设备
            test_data_y = test_data_y.to(device) # 将标签移动到指定设备

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y).item() # 修正：使用 .item() 获取标量值
            test_num += test_data_x.size(0)

    test_acc = test_corrects / test_num # 计算准确率
    print("测试准确率: " + str(test_acc)) # 修正：将浮点数转换为字符串再拼接

if __name__=="__main__":
    # 加载模型
    model = ResNet(Residual)
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    # 利用现有的模型进行模型的测试
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    # device = 'cpu'
    # model = model.to(device)
    # classes = ['苹果', '香蕉', '葡萄', '橘子', '梨']
    # with torch.no_grad():
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #
    #         # 设置模型为验证模型
    #         model.eval()
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output, dim=1)
    #         result = pre_lab.item()
    #         label = b_y.item()
    #         print("预测值：", classes[result], "------", "真实值：", classes[label])

#     image = Image.open('img.png')
#     if image.mode == 'RGBA':
#         image = image.convert('RGB')
#     normalize = transforms.Normalize([0.22890999, 0.1963964, 0.14335695], [0.09950233, 0.07996743, 0.06593084])
#     # 定义数据集处理方法变量
#     test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
#     image = test_transform(image)

#     # 添加批次维度
#     image = image.unsqueeze(0)

#     with torch.no_grad():
#         model.eval()
#         image = image.to(device)
#         output = model(image)
#         pre_lab = torch.argmax(output, dim=1)
#         result = pre_lab.item()
#     print("预测值：", classes[result])