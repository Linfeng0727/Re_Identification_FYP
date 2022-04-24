import wx
import os
import cv2


class ReidUI(wx.Frame):
    def __init__(self, *args, **kw):
        super(ReidUI, self).__init__(*args, **kw)
        self.video_size = (490, 280)
        self.input_sample_size = (42, 121)
        self.video_path = 'dark.jpg'
        self.video_cover = wx.Image(self.video_path, wx.BITMAP_TYPE_ANY).Scale(self.video_size[0], self.video_size[1])

        self.Input_Path = 'dark.jpg'
        self.Input_Img = wx.Image(self.Input_Path, wx.BITMAP_TYPE_ANY).Scale(self.input_sample_size[0],
                                                                             self.input_sample_size[1])

        self.init_ui()

    def init_ui(self):
        # Generation of Menubar on the top of the app
        self.createmenubar()

        # Generation of Panel1 and Components
        pnl1 = wx.Panel(self)
        pnl1.SetBackgroundColour(wx.BLACK)
        self.bmp = wx.StaticBitmap(pnl1, -1, wx.Bitmap(self.video_cover))

        # Generation of panel2 and Components
        pnl2 = wx.Panel(self)
        # slider1 = wx.Slider(pnl2, value=0, minValue=0, maxValue=1000)
        bmp1 = wx.ArtProvider.GetBitmap(wx.ART_FOLDER_OPEN, wx.ART_OTHER, (32, 32))
        bmp2 = wx.ArtProvider.GetBitmap(wx.ART_GO_FORWARD, wx.ART_OTHER, (32, 32))
        bmp3 = wx.ArtProvider.GetBitmap(wx.ART_FIND, wx.ART_OTHER, (32, 32))
        bmp4 = wx.ArtProvider.GetBitmap(wx.ART_NORMAL_FILE, wx.ART_OTHER, (32, 32))
        bmp5 = wx.ArtProvider.GetBitmap(wx.ART_DELETE, wx.ART_OTHER, (32, 32))

        open_video = wx.BitmapButton(pnl2, bitmap=bmp1)
        open_video.Bind(wx.EVT_BUTTON, self.open_video)

        play = wx.BitmapButton(pnl2, bitmap=bmp2)
        play.Bind(wx.EVT_BUTTON, self.play_film)

        reid = wx.BitmapButton(pnl2, bitmap=bmp3)
        reid.Bind(wx.EVT_BUTTON, self.reid_process)

        input_sample = wx.BitmapButton(pnl2, bitmap=bmp4)
        input_sample.Bind(wx.EVT_BUTTON, self.input_sample)

        delete_all = wx.BitmapButton(pnl2, bitmap=bmp5)
        delete_all.Bind(wx.EVT_BUTTON, self.delete_all)

        # h_box3
        self.slider2 = wx.StaticBitmap(pnl2, -1, wx.Bitmap(self.Input_Img))
        label = wx.StaticText(pnl2, -1, 'Sample Inputted')

        # Generation of BoxSizer
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox1 = wx.BoxSizer(wx.HORIZONTAL)
        h_box2 = wx.BoxSizer(wx.HORIZONTAL)
        h_box3 = wx.BoxSizer(wx.VERTICAL)

        h_box2.Add(open_video, flag=wx.LEFT, border=10)
        h_box2.Add(input_sample, flag=wx.RIGHT, border=50)

        h_box2.Add(reid, flag=wx.LEFT, border=50)
        h_box2.Add(play)

        h_box2.Add((-1, -1), proportion=1)
        h_box2.Add(delete_all)
        # h_box2.Add(slider2, flag=wx.TOP|wx.LEFT, border=5)

        h_box3.Add(label, flag=wx.RIGHT, border=5)
        h_box3.Add(self.slider2, flag=wx.LEFT, border=25)

        # vbox.Add(h_box1, flag=wx.RIGHT, border=100)
        vbox.Add(h_box2, proportion=1, flag=wx.TOP, border=50)

        vbox1.Add(vbox, flag=wx.EXPAND | wx.RIGHT, border=50)
        vbox1.Add(h_box3, flag=wx.EXPAND | wx.RIGHT)

        pnl2.SetSizer(vbox1)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(pnl1, proportion=1, flag=wx.EXPAND)
        sizer.Add(pnl2, flag=wx.EXPAND | wx.BOTTOM | wx.TOP, border=10)

        self.SetMinSize((500, 500))
        self.SetMaxSize((500, 500))
        self.CreateStatusBar()
        self.SetSizer(sizer)
        self.SetSize((500, 500))
        self.SetTitle('Player')
        self.Centre()
        self.Show(True)

    def createmenubar(self):
        # Initialization of MenuBar
        menubar = wx.MenuBar()

        # Options in this menubar
        file = wx.Menu()
        file.Append(101, '&quit', 'Quit Application')
        file.Append(102, '&open', 'Open the file')
        file.Bind(wx.EVT_MENU, self.file_menu)

        play = wx.Menu()
        play.Append(201, '&run', 'Play the Video')
        play.Bind(wx.EVT_MENU, self.play_menu)

        tools = wx.Menu()
        tools.Append(301, '&reid', 'Process the video')
        tools.Bind(wx.EVT_MENU, self.tools_menu)

        menubar.Append(file, '&File')
        menubar.Append(play, '&Play')
        menubar.Append(tools, '&Tools')

        self.SetMenuBar(menubar)

    def file_menu(self, evt):
        evt_id = evt.GetId()
        if evt_id == 101:
            self.Close()
        elif evt_id == 102:
            self.open_video(self)

    def play_menu(self, evt):
        evt_id = evt.GetId()
        if evt_id == 201:
            self.play_film(self)

    def tools_menu(self, evt):
        evt_id = evt.GetId()
        if evt_id == 301:
            self.reid_process(self)

    def open_video(self, evt):
        wildcard = 'All files(*.*)|*.*'
        dialog = wx.FileDialog(None, 'select', os.getcwd(), '', wildcard, wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.video_path = dialog.GetPath()
            self.video_cover = wx.Image('play.jpeg', wx.BITMAP_TYPE_ANY).Scale(
                self.video_size[0], self.video_size[1])
            self.bmp.SetBitmap(wx.Bitmap(self.video_cover))
        dialog.Destroy

    def play_film(self, evt):
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            print("Fail to open video！")

        while True:
            _, frame = capture.read()
            if frame is None:
                break

            cv2.waitKey(30)  # The pause time
            img = cv2.resize(frame, (self.video_size[0], self.video_size[1]))
            image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pic = wx.Bitmap.FromBuffer(self.video_size[0], self.video_size[1], image1)
            self.bmp.SetBitmap(wx.Bitmap(pic))

        capture.release()

    def reid_process(self, evt):
        pass

    def input_sample(self, evt):
        wildcard = 'All files(*.*)|*.*'
        dialog = wx.FileDialog(None, 'select', os.getcwd(), '', wildcard, wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.Input_Path = dialog.GetPath()
            self.Input_Img = wx.Image(self.Input_Path, wx.BITMAP_TYPE_ANY).Scale(
                self.input_sample_size[0], self.input_sample_size[1])
            self.slider2.SetBitmap(wx.Bitmap(self.Input_Img))
        dialog.Destroy

    def delete_all(self, evt):
        self.video_path = 'dark.jpg'
        self.Input_Path = 'dark.jpg'
        self.video_cover = wx.Image(self.video_path, wx.BITMAP_TYPE_ANY).Scale(self.video_size[0], self.video_size[1])
        self.bmp.SetBitmap(wx.Bitmap(self.video_cover))
        self.Input_Img = wx.Image(self.Input_Path, wx.BITMAP_TYPE_ANY).Scale(self.input_sample_size[0],
                                                                             self.input_sample_size[1])
        self.slider2.SetBitmap(wx.Bitmap(self.Input_Img))


if __name__ == '__main__':
    ex = wx.App()
    ReidUI(None)
    ex.MainLoop()
