import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

from deep_q_netwok import DeepQNetwork

from kivy.core.window import Window
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# last_x ve last_y haritaya kumu çizdiğimiz son koordinatlardır
last_x = 0
last_y = 0
# n_points en son kum çizimizdeki toplam nokta sayısı
n_points = 0
# length çizdiğimiz kumun uzunluğu
length = 0

brain = DeepQNetwork(5,3,0.9) # Burada ağımıza 5 sensör 3 aksyion ve gama değerini 0.9 olarak veriyoruz
# action 0 => dönme action 1 => 20 derece dön action 2 => -20 derece dön
action2rotation = [0,20,-20] 
# ajanın alacağı ödülleri initialize ediyoruz
last_reward = 0
# Zamana göre ortalama puan eğrisi
scores = []

first_update = True
def init():
    global sand # kum bir arraydir. grafik arayüzümüzde piksellere tekabül eder. 1 kum var 0 yok demektir.
    global goal_x # ajanın hedefinin x koordinatı
    global goal_y # ajanın hedefinin y koordinatı
    global first_update 
    sand = np.zeros((longueur,largeur)) # kum arrayimizi 0 lar ile oluşturuyoruz
    goal_x = 20 # ajanın hedefinin x koordinatı bu değer soldan 20px uzağa denk gelir
    goal_y = largeur - 20 # ajanın hedefinin y koordinatı bu değer sol üstten 20px uzağa denk gelir
    first_update = False


last_distance = 0


# NumericProperty() Bu özellik, bir değerin bir sayısal değeri temsil etmesi gerektiğini belirten bir özel bir özelliktir. (0) aldığı değer değişkenin ilk değeridir
# ReferenceListProperty
# Bu özellik birden fazla özelliği içeren bir liste temsil etmek için kullanılır. 
# genellikle bir nesnenin çeşitli özelliklerinin bir arada kullanılması gerektiği durumlarda kullanışlıdır. 
# örneğin bir nesnenin x ve y koordinatlarını aynı anda ele almak için kullanılabilir.

class Car(Widget):
    
    angle = NumericProperty(0) # Araba ajanımızın açısı
    rotation = NumericProperty(0) # Arabanın son döndüğü nokta bu aksiyonla birlikte 0 20 ve -20 değerlerinden birini alacak
    velocity_x = NumericProperty(0) # Aracın hız vektörünün x koordinatı
    velocity_y = NumericProperty(0) # Aracın hız vektörünün y koordinatı
    velocity = ReferenceListProperty(velocity_x, velocity_y) # Aracın Hız vektörü
    sensor1_x = NumericProperty(0) # Aracın 1. sensörünün x koordinatı
    sensor1_y = NumericProperty(0) # Aracın 1. sensörünün y koordinatı
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y) # Aracın 1. sensör vektörü
    sensor2_x = NumericProperty(0) 
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0) # Sinyal 1 bu sinyal 1. sensörden alınacak
    signal2 = NumericProperty(0) # Sinyal 2 bu sinyal 2. sensörden alınacak
    signal3 = NumericProperty(0) # Sinyal 3 bu sinyal 3. sensörden alınacak

    def move(self, rotation):
        # Aracın pozisyonu aracın son hız vektörüne ve son pozisyonuna göre güncellenecek
        self.pos = Vector(*self.velocity) + self.pos 
        # Aracın son rotasyonunu alıyoruz
        self.rotation = rotation
        # Aracın açısını güncelliyoruz
        self.angle = self.angle + self.rotation
        # Sensör 1 in pozisyonunu ayarlıyoruz bu sensör ileriye bakacak
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        # Sensör 2 nin pozisyonunu ayarlıyoruz bu sensör +30 derece ile sola bakacak
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        # Sensör 3 ün pozisyonunu ayarlıyoruz bu sensör -30% derece ile sağa bakacak
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos

        # Sensör 1 den gelen sinyali alıyoruz.
        # Burada sinyal sensör etrafındaki kum yoğunluğu
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        # Sensör 2 den gelen sinyali alıyoruz
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        # Sensör 3 den gelen sinyali alıyoruz
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.

        # Bu if blokları aracın mapin dışına çıkıp çıkmadığını kontrol ediyorlar
        # Eğer aracın sensörlerinden biri aşağıdaki koşulları sağlıyorsa aracın mapin kenarlarında gezindiğini anlıyoruz
        # Aracın mapin dışına çıkmasını engellemek için aracın mapin dışına geldiği açıdaki sensörüne kum yoğunluğunun 1 (En Fazla) olduğunu gönderiyoruz
        # Bu sayede araç kum yoğunluğuna değer alacağı - değerli ödülü almamak için mapin dışına çıkamıyor.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget): # Sensör 1i temsil eden obje
    pass
class Ball2(Widget): # Sensör 2yi temsil eden obje
    pass
class Ball3(Widget): # Sensör 3ü temsil eden obje
    pass

# ObjectProperty, Kivy framework içinde kullanılan bir özelliktir. Bu özellik, bir nesne örneğini temsil eder 
# Genellikle bir Widget'in başka bir Widget veya Kivy objesi ile ilişkilendirilmesi gerektiğinde kullanılır. 

class Game(Widget):
    
    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        

    car = ObjectProperty(None) # Kivy dosyamızdan araba objemizi alıyoruz
    ball1 = ObjectProperty(None) # Kivy dosyamızdan sensör 1 için ball değişkenimizi alıyoruz
    ball2 = ObjectProperty(None) # Kivy dosyamızdan sensör 2 için ball değişkenimizi alıyoruz
    ball3 = ObjectProperty(None) # Kivy dosyamızdan sensör 3 için ball değişkenimizi alıyoruz

    # Bu kod uygulamayı çalıştıdığımızda aracı başlatacak
    def serve_car(self):
        self.car.center = self.center # Araç mapin ortasında başlayacak
        self.car.velocity = Vector(6, 0) # Araç sağa doğru 6 hızla başlatılacak

    # Yeni bir duruma ulaşıldığında (Sensörlerden her yeni sinyal alındığında) her t zamanında güncellenmesi gereken herşeyi güncelleyen fonksiyon
    def update(self, dt):

        global brain # Deep Q Networkümüz
        global last_reward # Son ödülümüz
        global scores # Ödüllerimizin ortalaması 
        global last_distance # Araç ve hedef arasındaki son uzaklığımız
        global goal_x # Hedefin x koordinatı
        global goal_y # Hedefin y koordinatı
        global longueur # mapin genişliği
        global largeur # mapin yüksekliği

        longueur = self.width
        largeur = self.height
        if first_update: # mapi 1 kez başlatma kodu 
            init()
            self.painter = MyPaintWidget()

        xx = goal_x - self.car.x # araç ile hedef arasındaki x koordinatlarının farkı
        yy = goal_y - self.car.y # araç ile hedef arasındaki y koordinatlarının farkı
        # Aracın hedefe göre yönlendirilmesi. Eğer araç hedefe tam olarak doğru gidiyorsa o zaman orientation = 0 olacaktır.
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        # Üç sensör tarafından alınan üç sinyal ve yönlendirme ve yönlendirmenin tersini aldığımız giriş vektörümüz
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        # Verdiğimiz verilerle networkümüz üzerinde bir aksiyon alıyor ve bunu değişkene atıyoruz
        action = brain.update(last_reward, last_signal)
        # Scorlarımıza networkümüzden dönen skorları ekliyoruz
        scores.append(brain.score())
        # Networkümüzün verdiği aksiyona göre yeni rotasyonumuzu belirliyoruz 0, +20 derece, -20 derece olabilir.
        rotation = action2rotation[action]
        # Ve networkümüz yardımıyla belirlediğimiz rotasyona göre ajanımızı hareket ettiriyoruz
        self.car.move(rotation)
        # Aracın hareketinden sonra araç ve hedef arasındaki uzaklığı tekrar hesaplıyoruz
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        # Burada aracın hareketinden sonra sensörlerimizin konumlarını da aracımıza göre güncelliyoruz
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        # Bu if aracın kumun üzerinde olu olmadığını sorgular.
        if sand[int(self.car.x),int(self.car.y)] > 0: # Eğer araç kumun üzerindeyse :
            # Aracın hızını 1 e indir
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            # Ajana -1 ödül ver
            last_reward = -1
        else: # Eğer araç kumun üzerinde değilse :
            # Aracın hızını değiştirme
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            # Araca -0.2 ödül ver (Buna living penalty denir)
            # Living penalty ajan hedefe doğru yaklaşmadığı durumlarda aracın hedefe yönelimini teşvik etmek amacıyla kullanılır
            last_reward = -0.2
            if distance < last_distance:  # Eğer ajanımız hedefe yaklaştıysa
                # Araca 0.1 ödül ver.
                # Bu ajanımızın hedefe doğru giden yolların Q değerlerini daha yüksek atayarak bu yolları tercih etmesi gerektiğini öğretir
                last_reward = 0.1 

        # Bu if blokları hareket sonrarı aracın mapin herhangi bir köşesine çarpıp çarpmadığını kontrol eder
        # Eğer araç herhangi bir köşeye çarptıysa : 
        # Aracın konumunu düzenle
        # Ve ona -1 ödül ver
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        # Bu kod aracın hedefe ulaşıp ulaşmadığını kontrol eder
        # Eğer araç hedefe ulaştıysa hedefin yerini değiştirir
        if distance < 100: # Eğer araç hedefe ulaştıysa 
            goal_x = self.width-goal_x # Hedefin x koordinatını sağ alt köşeye taşı
            goal_y = self.height-goal_y # Hedefin y koordinatını sağ alt köşeye taşı
        
        # Bu kod hareket sonrası aracın hedefe olan uzaklığını kaydeder
        last_distance = distance

# Adding the painting tools
from kivy.core.window import Window

class MyRandomSandWidget(Widget):
    def __init__(self, **kwargs):
        super(MyRandomSandWidget, self).__init__(**kwargs)
        self.sand_circles = []
        

        # Schedule sand circle generation and clearing
        Clock.schedule_interval(self.generate_sand_circle, 2)
        Clock.schedule_interval(self.draw_sand_circles, 2.1)
        Clock.schedule_interval(self.clear_sand_circles, 4)

    def generate_sand_circle(self, dt):
        global sand
        x = np.random.randint(0, Window.width)
        y = np.random.randint(0, Window.height)
        sand[x,y] = 1
        self.sand_circles.append((x, y))

    def clear_sand_circles(self, dt):
        global sand
        for circle in self.sand_circles:
            x, y = circle
            sand[x,y] = 0
            with self.canvas:
                Color(0, 0, 0)  # Set color to black for clearing
                Ellipse(pos=(x - 10, y - 10), size=(20, 20))

        self.sand_circles = []  # Clear the list

    def draw_sand_circles(self,dt):
        with self.canvas:
            Color(0.8, 0.7, 0)
            for circle in self.sand_circles:
                x, y = circle
                Ellipse(pos=(x - 10, y - 10), size=(20, 20))

class MyPaintWidget(Widget):
    def __init__(self, **kwargs):
        super(MyPaintWidget, self).__init__(**kwargs)
        self.sand_circles = []
        

        # Schedule sand circle generation and clearing
        Clock.schedule_interval(self.generate_sand_circle, 2)
        Clock.schedule_interval(self.draw_sand_circles, 2.1)
        Clock.schedule_interval(self.clear_sand_circles, 4)

    def generate_sand_circle(self, dt):
        global sand
        x = np.random.randint(0, Window.width)
        y = np.random.randint(0, Window.height)
        sand[x,y] = 1
        self.sand_circles.append((x, y))

    def clear_sand_circles(self, dt):
        global sand
        for circle in self.sand_circles:
            x, y = circle
            sand[x,y] = 0
            with self.canvas:
                Color(0, 0, 0)  # Set color to black for clearing
                Ellipse(pos=(x - 10, y - 10), size=(20, 20))

        self.sand_circles = []  # Clear the list

    def draw_sand_circles(self,dt):
        with self.canvas:
            Color(0.8, 0.7, 0)
            for circle in self.sand_circles:
                x, y = circle
                Ellipse(pos=(x - 10, y - 10), size=(20, 20))
    # Bu kod mouse sol tuşuna basıldığında mape biraz kum bırakır
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    # Bu kod mouse sol tuşuna basılı tutulup mapin üzerinde gezdirildiğinde mouse hattı boyunca kum bırakır
    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y
        

class CarApp(App):

    # Uygulamayı build eden kod
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    # Bu kod mapimiz üzerindeki kumları temizlemek için kullanacağımız butonun kodları
    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    # Bu kod ajanımızın o zamana kadar geliştirdiği policy'i kaydeder ve konsola tekrara göre puanın grafiğini çıkarır
    def save(self, obj):
        print("Policy Kaydediliyor...")
        brain.save()
        plt.plot(scores)
        plt.show()
    # Son kaydettiğimiz policynin ajana yüklenerek o policynin üzerinde aksiyon aldıran buton.
    def load(self, obj):
        print("Son Kaydedilen Policy Yükleniyor...")
        brain.load()

if __name__ == '__main__':
    CarApp().run()
