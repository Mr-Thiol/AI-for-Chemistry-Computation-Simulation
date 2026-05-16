import yaml
from easydict import EasyDict

class Config(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        elif type(d) == str:
            with open(d,'r') as f:
                d = yaml.safe_load(f)
        else:
            d = dict(d)        
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update','to_dict', 'pop','fill','fill_default','save'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)   # 把所有的dict都改为config类。所以在保存的时候会有一大堆奇怪的方法被保存
        super(Config, self).__setattr__(name, value)
        super(Config, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(Config, self).pop(k, d)
    
    # load default settings
    def fill(self,current_dict=None,fill_dict=None):        
        # update
        for key, fill_value in fill_dict.items():
            if type(key) == str and (key not in current_dict.keys()): # if not exists key, direct load
                current_dict[key] = fill_value
                            
            elif isinstance(fill_value, EasyDict) and isinstance(current_dict[key], Config): # get deeper
                self.fill(current_dict[key], fill_value)
    
    @classmethod
    def to_dict(cls,easydict_obj=None):
        if isinstance(easydict_obj, Config):
            # 将EasyDict对象转化为普通的dict
            dict_obj = easydict_obj.__dict__.copy()    

            # 递归处理嵌套的EasyDict
            for key, value in dict_obj.items():
                if isinstance(value, Config):
                    dict_obj[key] = cls.to_dict(value)    

            return dict_obj
        elif isinstance(easydict_obj, (list, tuple)):
            # 如果是列表或元组，递归处理其中的每个元素
            return [cls.to_dict(item) for item in easydict_obj]
        else:
            return easydict_obj
    
    def save(self, path):
        # 需要把奇怪的类剔除掉，防止保存太多奇怪的东西，只保存str,float,int 这样的数据结构就行了
        output_dict = self.to_dict(self)
        with open(path, 'w') as f:
            yaml.safe_dump(output_dict, f, default_flow_style=False)
            
if __name__ == '__main__':
    config = Config()
    config.fill_default(config)
    config.save('default.yml')