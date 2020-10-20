r"""Functional interface"""
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, classification_report

__call__ = ['Accuracy', 'AUC', 'F1Score', 'EntityScore', 'ClassReport', 'MultiLabelReport', 'AccuracyThresh']

all_labels = ['为何召回快递', '主播推荐', '优惠券无法领', '使用及教学视频', '使用周期', '保修卡咨询', '免息活动时间', '发送截图', '合作推广', '商品分类', '商品日期',
                '商品清单', '商品说明', '失望情绪', '安装异常', '安装网点', '安装费用', '尾款支付问题', '尾款无法支付', '延迟下单', '愤怒情绪', '无法下单', '无法分期付款',
                '是否免息', '是否备注', '显示有下单未付款交易', '求加微信', '没有效果', '直播优惠', '直播奖品', '直播渠道', '直播购买指南', '直播间福利', '直播领券',
                '索要说明书', '规格尺寸区别', '试用装咨询', '货源产地', '辱骂语言', '退货单缺失', '邮费差价', '高危风险', '下单减价', '下单备注', '交易关闭', '付款失败',
                '价格变动', '信用卡支付', '修改价格', '修改收货信息', '修改订单信息', '修改邮费', '加不了购物车', '加单咨询', '去拍去付', '取消订单', '合并付款', '如何下单付款',
                '如何下单选购', '定金支付问题', '定金无法支付', '已拍已付', '微信支付', '我要买', '拍错商品', '未下单付款', '未支付尾款', '确认收货时间', '能否下单付款',
                '补差价购买', '补邮费咨询', '订单状态未同步', '运费险购买', '重复付款', '重拍商品', '上门取件', '不喜欢不合适', '丢件', '为何拒收', '为何拒绝退款',
                '修改退款信息', '催促换货操作', '催促退款', '全国联保', '准备或者已经申请退款', '到付运费', '发错商品', '发错地址', '取消退款', '售后服务咨询', '售后服务电话',
                '售后维修网点', '商品质量问题', '坏了', '填改退货单号', '安装教程', '安装服务及费用', '少件', '影响退换因素', '换货条件', '换货维修运费', '换货需求', '描述不符',
                '无法申请退款', '无理由退换', '显示确认收货', '未收到货', '查询或下载发票', '漏了', '申请退款后发货', '破损', '维修需求', '补偿赔偿', '补发需求', '要求拒绝退款',
                '货已退回', '赠品缺失', '赠品质量问题', '退件查询', '退差价咨询', '退换货地址', '退换货物流', '退款关闭', '退款原因', '退款去向', '退款成功', '退款金额',
                '退货形式', '退货时效', '退货条件', '退货注意事项', '退货运费', '预约安装', '为何下架', '价格区别', '何时上货', '保价咨询', '保修咨询', '凑单咨询',
                '到手价咨询', '功效功能', '包装区别', '包装方式', '包装要求', '发售咨询', '发货数量', '商品价格', '商品单卖', '商品尺寸', '商品挑选更换', '商品材质',
                '商品规格', '搭配推荐', '新旧款区别', '是否有货', '是否正品', '是否预售', '清洁保养', '版本款型区别', '用法用量', '索要图片信息', '质量情况', '购买渠道区别',
                '适用人群', '邮费咨询', '配件套件单买', '颜色区别', '颜色推荐', '不要开票', '使用花呗', '修改发票', '分地址发货', '加卡片', '发票缺失', '增值税发票',
                '如何评价', '小票', '延长收货时间', '开票内容', '开票时间', '开票类型', '投诉需求', '支持花呗', '普通发票', '礼品包装', '花呗期付', '花呗还款', '花呗额度',
                '订单信息查询', '货到付款', '货票同行', '赠品挑选更换', '赠送运费险', '运费险理赔', '返现咨询', '送货形式', '邮寄发票', '重开发票', '门店信息', '中奖名单',
                '为何还有运费', '优惠券如何用', '优惠券无法用', '优惠券未使用', '优惠券未收到', '优惠券退款', '优惠叠加', '优惠影响', '会员信息查询', '会员申请', '免定金咨询',
                '商品议价', '好评返利', '宝贝为何原价', '店铺活动', '抢不到', '换购', '活动奖品', '活动时间', '直播价格', '秒杀', '积分兑换', '红包使用', '红包无法用',
                '红包未收到', '红包未用', '红包退款', '能否多送', '能否活动价格购买', '能否直播价购买', '能否降低活动门槛', '观看直播', '请求活动通知', '请求活动链接', '购买限制',
                '购物券使用', '购物券无法用', '购物券未使用', '购物券未收到', '购物券退款', '赠品', '赠品属性咨询', '返红包', '邮费议价', '预售尾款', '预售退款', '领取优惠券',
                '领取红包', '领取购物券', '不发要退货', '保密发货', '催促发货', '催促快递', '发货地', '发货形式', '寄件范围', '指定快递', '提醒发货检查', '无法拒收',
                '无物流记录', '显示签收', '未收到但显示已签收', '查询物流', '物流单号咨询', '物流异常', '物流揽件中', '物流无更新', '空包', '索要取件码', '索要换货信息',
                '索要补发信息', '索要重发信息', '联系快递', '要求召回快递', '要求延迟发货', '货已拒收', '货已收到', '赠品发货咨询', '通常到货时间', '通常发货时间', '预约送货时间',
                '默认快递', '否认', '希望', '帮忙', '感叹', '放弃', '期待', '求回复', '求等', '谅解', '难过']


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class Accuracy(Metric):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    Examples:
        >>> metric = Accuracy(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''

    def __init__(self, topK):
        super(Accuracy, self).__init__()
        self.topK = topK
        self.reset()

    def __call__(self, logits, target):
        _, pred = logits.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        self.correct_k = correct[:self.topK].view(-1).float().sum(0)
        self.total = target.size(0)

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        return float(self.correct_k) / self.total

    def name(self):
        return 'accuracy'


class AccuracyThresh(Metric):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    Example:
        >>> metric = AccuracyThresh(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''

    def __init__(self, thresh=0.5):
        super(AccuracyThresh, self).__init__()
        self.thresh = thresh
        self.reset()

    def __call__(self, logits, target):
        self.y_pred = logits.sigmoid()
        self.y_true = target

    def reset(self):
        self.correct_k = 0
        self.total = 0

    def value(self):
        data_size = self.y_pred.size(0)
        acc = np.mean(((self.y_pred > self.thresh) == self.y_true.byte()).float().cpu().numpy(), axis=1).sum()
        return acc / data_size

    def name(self):
        return 'accuracy'

    def classification_report(self):
        self.y_pred = self.y_pred > self.thresh
        report = classification_report(self.y_true.numpy(),
                                       self.y_pred.numpy(),
                                       target_names=all_labels)
        print(report)


class AUC(Metric):
    '''
    AUC score
    micro:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = AUC(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''

    def __init__(self, task_type='binary', average='binary'):
        super(AUC, self).__init__()

        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']

        self.task_type = task_type
        self.average = average

    def __call__(self, logits, target):
        '''
        计算整个结果
        '''
        if self.task_type == 'binary':
            self.y_prob = logits.sigmoid().data.cpu().numpy()
        else:
            self.y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def value(self):
        '''
        计算指标得分
        '''
        auc = roc_auc_score(y_score=self.y_prob, y_true=self.y_true, average=self.average)
        return auc

    def name(self):
        return 'auc'


class F1Score(Metric):
    '''
    F1 Score
    binary:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
    micro:
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
    macro:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
    weighted:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
    samples:
            Calculate metrics for each instance, and find their average.
    Example:
        >>> metric = F1Score(**)
        >>> for epoch in range(epochs):
        >>>     metric.reset()
        >>>     for batch in batchs:
        >>>         logits = model()
        >>>         metric(logits,target)
        >>>         print(metric.name(),metric.value())
    '''

    def __init__(self, thresh=0.5, normalizate=True, task_type='binary', average='binary', search_thresh=False):
        super(F1Score).__init__()
        assert task_type in ['binary', 'multiclass']
        assert average in ['binary', 'micro', 'macro', 'samples', 'weighted']

        self.thresh = thresh
        self.task_type = task_type
        self.normalizate = normalizate
        self.search_thresh = search_thresh
        self.average = average

    def thresh_search(self, y_prob):
        '''
        对于f1评分的指标，一般我们需要对阈值进行调整，一般不会使用默认的0.5值，因此
        这里我们队Thresh进行优化
        :return:
        '''
        best_threshold = 0
        best_score = 0
        for threshold in tqdm([i * 0.01 for i in range(100)], disable=True):
            self.y_pred = y_prob > threshold
            score = self.value()
            if score > best_score:
                best_threshold = threshold
                best_score = score
        return best_threshold, best_score

    def __call__(self, logits, target):
        '''
        计算整个结果
        :return:
        '''
        self.y_true = target.cpu().numpy()
        if self.normalizate and self.task_type == 'binary':
            y_prob = logits.sigmoid().data.cpu().numpy()
        elif self.normalizate and self.task_type == 'multiclass':
            y_prob = logits.softmax(-1).data.cpu().detach().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type == 'binary':
            if self.thresh and self.search_thresh == False:
                self.y_pred = (y_prob > self.thresh).astype(int)
                self.value()
            else:
                thresh, f1 = self.thresh_search(y_prob=y_prob)
                print(f"Best thresh: {thresh:.4f} - F1 Score: {f1:.4f}")
                return thresh, f1

        if self.task_type == 'multiclass':
            self.y_pred = np.argmax(y_prob, 1)

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        '''
         计算指标得分
         '''
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        return f1

    def name(self):
        return 'f1'


class ClassReport(Metric):
    '''
    class report
    '''

    def __init__(self, target_names=None):
        super(ClassReport).__init__()
        self.target_names = target_names

    def reset(self):
        self.y_pred = 0
        self.y_true = 0

    def value(self):
        '''
        计算指标得分
        '''
        score = classification_report(y_true=self.y_true,
                                      y_pred=self.y_pred,
                                      target_names=self.target_names)
        print(f"\n\n classification report: {score}")

    def __call__(self, logits, target):
        _, y_pred = torch.max(logits.data, 1)
        self.y_pred = y_pred.cpu().numpy()
        self.y_true = target.cpu().numpy()

    def name(self):
        return "class_report"


class MultiLabelReport(Metric):
    '''
    multi label report
    '''

    def __init__(self, id2label=None):
        super(MultiLabelReport).__init__()
        self.id2label = id2label

    def reset(self):
        self.y_prob = 0
        self.y_true = 0

    def __call__(self, logits, target):

        self.y_prob = logits.sigmoid().data.cpu().detach().numpy()
        self.y_true = target.cpu().numpy()

    def value(self):
        '''
        计算指标得分
        '''
        for i, label in self.id2label.items():
            auc = roc_auc_score(y_score=self.y_prob[:, i], y_true=self.y_true[:, i])
            print(f"label:{label} - auc: {auc:.4f}")

    def name(self):
        return "multilabel_report"
