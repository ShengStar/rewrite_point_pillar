from train import bulid_net
import torch
if __name__ == '__main__':
    filename = '000001.bin'
    net = bulid_net()
    net.load_state_dict(torch.load("voxelnet-296960.tckpt"))
    #print(net)
    #torch.save(net.state_dict(), 'parameter.pkl')
    #net.load_state_dict(torch.load('voxelnet-232000.tckpt'))
    #net.cuda()
    net.eval()
    preds_dict = net(filename)
    box_preds = preds_dict["box_preds"]
    cls_preds = preds_dict["cls_preds"]
    #print(x.shape)
    #net.load_state_dict(torch.load('parameter.pkl'))

