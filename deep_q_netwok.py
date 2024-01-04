import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Nöral ağımızın yapısını oluşturduğumuz class
class Network(nn.Module):

    def __init__(self, input_size, output_size):
        # Inherit ettiğimiz classın bütün toolarını kullanabilmemiz için gerekli olan kod.
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Networkumuzdeki bağlantıları ve nöron sayılarını belirliyoruz.
        # Burada networkümüz 5 => 30 => 30 => 3 Olacak şekilde ayarlandı.
        self.full_connection_one = nn.Linear(input_size,30)
        self.full_connection_two = nn.Linear(30, 30)
        self.full_connection_three = nn.Linear(30,output_size)

    # Bu fonksiyon ağın üzerinde aktivasyon fonksiyonlarını nöronlara uygulayarak ilerler
    # Ve tahmin edilen q_valueları döndürür
    def forward(self, state):
        hidden_neurons_first = F.relu(self.full_connection_one(state))
        hidden_neurons_second = F.relu(self.full_connection_two(hidden_neurons_first))
        q_values = self.full_connection_three(hidden_neurons_second)
        return q_values

# Experience Replay 
# Ajanımızın karşılaştığı ve öğrendiği şeyleri tutan bir hazfıza kodlayacağız.
# Buradaki deneyimler {state,action,reward} şeklinde tuple halinde tutulacak
class ReplayBuffer(object):

    def __init__(self, capaticy):
        self.capacity = capaticy # Replay Bufferimizin kaydedebileceği maksimum transition sayısı 
        self.memory = [] # Ajanımızın deneyimlerinin kaydedileceği hafıza

    # Bu fonksiyon memory'e deneyim eklememizi sağlar
    def push(self, transition):
        self.memory.append(transition)
        # Bu kodda memorymizin kapasitesini aşıp aşmadığını kontrol ediyoruz
        if len(self.memory) > self.capacity: # Eğer memory kapasitemizi aştıysa:
            # Memory içerisindeki en eski (dolayısıyla ilk) kaydı sil.
            del self.memory[0]

    # Bu fonksiyon memorymizin içerisinden rastgele batch_size kadar deneyim döndürür.
    def sample(self, batch_size):
        # zip(*) fonksiyonu : list = ((1,2,3),(4,5,6)) zip(* list) => ((1,4),(2,3),(5,6))
        # Bizim durumumuzda :
        # ((state1,action1,reward1),(state2,action2,reward2)) => ((state1,state2),(action1,action2),(reward1,reward2))
        # Bu teknik veriyi daha etkili bir şekilde kullanarak modelin öğrenme sürecini iyileştirmeye yardımcı olur.
        samples = zip(*random.sample(self.memory, batch_size))
        # Bu kod verilerimizi torch tensorlerine çevirir
        return map(lambda x : Variable(torch.cat(x, 0)), samples)
    
# Deep Q Learning

class DeepQNetwork():

    def __init__(self, input_size, number_of_actions, gamma):
        self.gamma = gamma
        # Son 100 rewardın ortalaması olacak şekilde reward_window tanımlıyoruz.
        self.reward_window = []

        # Deep Q Networkümüz için Network sınıfımızdan bir nesne oluşturuyoruz.
        self.model = Network(input_size,number_of_actions)

        # Deep Q Networkümüz için ReplayBufferimizi oluşturuyoruz.
        self.memory = ReplayBuffer(100000)

        # Modelimizin parametreleriyle birlikte bir optimizer oluşturduk (Adam => "Adaptive moment estimation")
        # Ve Modelimiz için learning rate'i 0.001 olarak belirledik.
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)

        # Aşağıda tanımlanan last_state, last_action, last_reward initialization değerleridir.
        # Pytorch tensorflow gibi derin öğrenme kütüphaneleri inputları düz vektörler olarak kabul etmezler
        # Kendilerine özel tensorler haline çevirip bir de batch'e denk gelen fake dimension ile input vektörlerimizi yeniden düzenlememiz lazım
        self.last_state = torch.Tensor(input_size).unsqueeze(0)

        self.last_action = 0

        self.last_reward = 0

    # Deep Q Networkümüzde en uygun actionu seçecek fonksiyondur.
    def select_action(self, state):
        # Variable kullanarak PyTorchun automatic differentiation özelliğini kullanarak trainable bir değişken haline getirir
        # volatile = True ise model tahmin yaparken bu değişkenlerin üzerinden gradyan(türev) hesaplamasını engelleyip tahmin yaparken modelin eğitilmesini durdurmak içindir
        # Deep Q Learningde modelimiz tahmin yaparken aynı zamanda eğitilmesini istemeyiz.
        # Bunun sebebi eğer model hem tahmin yaparken hem kendini eğitirse bu durumda model kendine karşı bir hedef belirlemiş olur
        # Bu da algoritmanın istikrarsızlaşmasına sebep olur.

        # Tempeture Parameter
        # Softmax fonksiyonu Q değerlerini olasılık dağılımına dönüştüren bir fonksiyondur. Bu dağılım her bir eylemin seçilme olasılığını temsil eder.
        # Fakat softmax çıktıları direkt olarak Q value olarak kullanılırsa bu dağılım sıkışmış yani belirli bir eylemin seçilme olasılığı diğerlerinden
        # Çok daha yüksek
        # Veya Çok daha düşük olur.
        # Bunun önüne geçebilmek için tempeture parameter kullanılır.
        # Tempeture Parameter büyük değerlerde softmax çıktılarını daha yumuşak hale getirir ve bir eylemin olasılığının çok daha fazla öne çıkmasını engeller.
        probabilities = F.softmax(self.model(Variable(state, volatile= True))*100) # Tempeture Parameter = 100
        # multinomial fonksiyonu belirli bir eylemin seçilme olasılığına göre rastgele bir eylem seçer
        action = probabilities.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # Bu kod bir hedef değer (target value) hesaplamak için kullanılır.
        # batch_state (anlık durum) için tüm aksiyonların Q değerlerini tahmin eder ve gather fonksiyonu ile seçilen aksiyonu alır.
        # unsqueeze(1) fonksiyonu ile normalde bize 2 boyutlu değer döneceğinden değer dönerken boyut uyumsuzluğundan doğacak hatadan kurtulduk
        # en sonda squeeze(1) ile de oluşturduğumuz fake dimensionlardan kurtularak orijinal vektörümüzü elde ettik.
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)


        # Next state için maksimum Q değerini almak için bu kodu kullanıyoruz.
        # detach() fonksiynu tensörün gradyan hesaplarını devre dışı bırakır
        # Çünkü next state'i öğrenmek için değil loss function hesaplamak için kullanacağız.
        next_outputs = self.model(batch_next_state).detach().max(1)[0]

        target = self.gamma*next_outputs + batch_reward

        # Bu kod Q learningin kalbi olan temporal difference lossu hesaplar
        temporal_difference_loss = F.smooth_l1_loss(outputs,target)

        # Bu kod her öğrenme adımından önce önceden hesaplanmış gradyanları temizler.
        # Bu sayede gradyanların birbirine karışması ve hatalı güncellemerin önüne geçilir
        self.optimizer.zero_grad()
        # Bu fonksiyon önceden hesaplanmış TDLoss değerine göre gradyanları hesaplar
        # Bu gradyanlar weightslere göre kayıp fonksiyonunu nasıl minimize edilmesi gerektiğini gösterir.
        # retain_graph = True ise ana gradyanların saklanmasın ve daha sonra tekrar kullanılmasını sağlar
        # Bu parametre PyTorch da kullanılan bir özelliktir ve karmaşık hesaplamarlda gradyanları doğru hesaplamak amacıyla kullanılır
        temporal_difference_loss.backward(retain_graph = True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # Burada last_state ve new_state imiz birer PyTorch tensörü olduğu için last_action ve last_reward da öyle olmalı
        # torch.Tensor fonksiyonuyla bunları tensör haline getiremeyiz çünkü bunlar sadece birer sayı
        # last_action int olduğu için onu int() içerisine alıp torch.LongTensor ile pytorch tensörüne çevirirken
        # last_reward float türünce olduğundan int() içerisine almadan pytorch tensörüne çevirdik.
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))

        action = self.select_action(new_state)

        # Burada networkümüz action ürettikten sonra network eğitimi gerçekleştirilcektir.
        # Burada memorymiz içerisinde tutulan deneyim sayısının en az 100 adet olması sorgulanıyor
        # Çünkü deep q learningde modelin öğreneceği inputlar replay bufferden rastgele alınır
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        
        # Burada last_action değişkenimizi ağımızın ürettiği action ile güncelliyoruz
        self.last_action = action
        # Burada last_stateimizi ajan tarafından gönderilen state ile güncelliyoruz
        self.last_state = new_state
        # Burada last_rewardımızı ajan tarafından gönderilen reward ile güncelliyoruz
        self.last_reward = reward

        self.reward_window.append(reward)

        # Burada reward_windowumuz içerisinde tutualn reward sayısını 1000 i geçerse en eski rewardı çıkaracak bir kod yazdık
        # Bunun sebebi ortalama olarak son 1000 rewardın ortalamasını almak istememizdir.
        # Alınmak istenen ortalamaya göre son kaç rewardın ortalamasının alınmak istediği değiştirilebilir.
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        return action
    
    # Bu fonksiyon ödüllerimizin ortalamasını döndürecektir.
    # + 1. yazılması ise 0 a bölme hatasından kurtulmak için bir yöntemdir
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window) + 1.)
    
    # Bu fonksiyon ağın şimdiye kadarki weightlerini ve optimizerini kaydeder. 
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),},
                    'last_policy.pth')

    # Bu fonksiyon kaydedilen policynin modele tekrar yüklenip kaydedilen noktadan tekrar devam etmesini sağlar.
    def load(self):
        if os.path.isfile('last_policy.pth'):
            print("=> Policy Yükleniyor...")
            checkpoint = torch.load('last_policy.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done !")
        else :
            print("Kaydedilmiş Policy bulunamadı...")