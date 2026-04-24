class CalibWindow:
	"""
	Separate OpenCV window with trackbars for live CALIB_PARAMS tuning.
	Opens when C is pressed, closes when C is pressed again.
	Automatically regenerates ROIs on any slider change.
	"""
	WIN = 'Calibration'

	def __init__(self):
		self.open = False
		self.active_cam = 'Right'  # tracks which cam's params are shown
		self._last_vals = {}

	def _cam_idx(self):
		return CAM_ORDER.index(self.active_cam)

	def toggle(self, cam_name: str):
		self.active_cam = cam_name
		if self.open:
			self.close()
		else:
			self._build()

	def _build(self):
		p = CALIB_PARAMS[self.active_cam]
		cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(self.WIN, 500, 280)

		# Each trackbar: (name, min, max, initial)
		# Multiply floats by 100 to use int trackbars
		cv2.createTrackbar('VP X',        self.WIN, p['vanishing_point'][0],  DISP_W,     lambda v: self._on_change())
		cv2.createTrackbar('VP Y',        self.WIN, p['vanishing_point'][1],  DISP_H,     lambda v: self._on_change())
		cv2.createTrackbar('Near Y',      self.WIN, p['near_y'],              DISP_H,     lambda v: self._on_change())
		cv2.createTrackbar('Far Y',       self.WIN, p['far_y'],               DISP_H,     lambda v: self._on_change())
		cv2.createTrackbar('Left X',      self.WIN, p['left_x'],              DISP_W,     lambda v: self._on_change())
		cv2.createTrackbar('Right X',     self.WIN, p['right_x'],             DISP_W,     lambda v: self._on_change())
		cv2.createTrackbar('Spots',       self.WIN, p['n_spots'],             20,          lambda v: self._on_change())
		cv2.createTrackbar('Rows',        self.WIN, p['n_rows'],              6,           lambda v: self._on_change())
		# fisheye_distortion: range -0.30 to +0.30, stored as int -30..30 (divide by 100)
		fisheye_int = int(p['fisheye_distortion'] * 100) + 30  # shift so 0 maps to 30
		cv2.createTrackbar('Fisheye x100', self.WIN, fisheye_int,             60,          lambda v: self._on_change())
		strength_int = int(p.get('perspective_strength', 1.0) * 100)
		cv2.createTrackbar('Persp x100', self.WIN, strength_int, 100, lambda v: self._on_change())
		self.open = True
		cv2.createTrackbar('Min Height', self.WIN, p.get('min_roi_height', 60), DISP_H, lambda v: self._on_change())
		print(f"[Calib] Window open for {self.active_cam}. Drag sliders to tune, C to close.")

	def _read(self) -> dict:
		fisheye_int = cv2.getTrackbarPos('Fisheye x100', self.WIN)
		return {
			'vanishing_point': (
				cv2.getTrackbarPos('VP X',    self.WIN),
				cv2.getTrackbarPos('VP Y',    self.WIN),
			),
			'near_y':             cv2.getTrackbarPos('Near Y',   self.WIN),
			'far_y':              cv2.getTrackbarPos('Far Y',    self.WIN),
			'left_x':             cv2.getTrackbarPos('Left X',   self.WIN),
			'right_x':            cv2.getTrackbarPos('Right X',  self.WIN),
			'n_spots':   max(1,   cv2.getTrackbarPos('Spots',    self.WIN)),
			'n_rows':    max(1,   cv2.getTrackbarPos('Rows',     self.WIN)),
			'fisheye_distortion': (fisheye_int - 30) / 100.0,
			'perspective_strength': cv2.getTrackbarPos('Persp x100', self.WIN) / 100.0,
			'min_roi_height': max(20, cv2.getTrackbarPos('Min Height', self.WIN)),
		}

	def _on_change(self):
		if not self.open:
			return
		try:
			params = self._read()
			# Guard against degenerate near_y == far_y
			if params['near_y'] == params['far_y']:
				return
			CALIB_PARAMS[self.active_cam] = params
			roi_params = {k: v for k, v in params.items() if k not in ('min_roi_height',)}
			new_rois = generate_rois(
				cam_id=self.active_cam,
				**roi_params,
			)
			if new_rois:
				rebuild_spot_masks(self.active_cam, new_rois)
		except Exception as e:
			print(f"[Calib] _on_change error: {e}")

	def tick(self):
		"""Call once per display loop iteration to keep the window alive."""
		if self.open:
			cv2.waitKey(1)

	def close(self):
		if self.open:
			try:
				cv2.destroyWindow(self.WIN)
			except Exception:
				pass
			self.open = False
			print(f"[Calib] Window closed. Final params for {self.active_cam}:")
			print(f"  CALIB_PARAMS['{self.active_cam}'] = {CALIB_PARAMS[self.active_cam]}")


calib_window = CalibWindow()
