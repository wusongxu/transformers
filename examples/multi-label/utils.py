import os
import sys
import csv
from sklearn.metrics import matthews_corrcoef, f1_score
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single

            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """See base class."""
        return ['为何召回快递', '主播推荐', '优惠券无法领', '使用及教学视频', '使用周期', '保修卡咨询', '免息活动时间', '发送截图', '合作推广', '商品分类', '商品日期',
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

    def get_one_hot(self, label):
        labels2idx = {l: i for i, l in enumerate(self.get_labels())}
        one_hot = [0 for _ in range(len(self.get_labels()))]
        for l in label:
            index = labels2idx[l]
            one_hot[index] = 1
        return one_hot

    def _read_tsv(self, input_file):
        """Reads a tab separated value file."""
        all_label = self.get_labels()
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                line = line.strip('\n')
                line = line.split('\t')
                # true_label = []
                # for i, token in enumerate(line):
                #     if '__label__' in token:
                #         true_label.append(token)
                #         continue
                #     else:
                #         line = ''.join(line[i:])
                #     break
                # t = self.get_one_hot(true_label)
                # t = self.get_one_hot(line[0].index(all_label))
                t = [0 for _ in range(len(all_label))]
                t[all_label.index(line[0])] = 1
                lines.append([line[1], t])
            return lines

class MyProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.txt")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "valid")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = None
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        elif output_mode == 'multi_label':
            label_id = example.label
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % (str(label_id)))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return {"acc_and_f1": acc_and_f1(preds, labels)}


output_modes = {
    "MyProcessor": "multi_label",
}

processors = {
    "MyProcessor": MyProcessor,
}
