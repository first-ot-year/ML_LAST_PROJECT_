from manim import *
import random
import os
import numpy as np



class IntroScene(Scene):
    def construct(self):

        self.camera.background_color = "#0a0a0a"
        self.add_sound("audio/proyecto4/Intro_p4.wav")

        color_izq = "#00ffff"
        color_der = "#00ffff"
        color_cen = "#ff66cc"

        # Edificio Izquierdo
        edificio_izq = Rectangle(width=2.2, height=4, color=color_izq, stroke_width=4, fill_opacity=0.1,
                                 fill_color=color_izq).shift(LEFT * 4.5)
        pisos_izq = VGroup(*[
            Line(edificio_izq.get_left() + UP * y, edificio_izq.get_right() + UP * y, color=color_izq, stroke_width=3)
            for y in np.linspace(
                edificio_izq.get_bottom()[1],
                edificio_izq.get_top()[1],
                5  # 4 pisos → 5 puntos
            )[1:-1]
        ])

        # Edificio Derecho
        edificio_der = Rectangle(width=2.2, height=4, color=color_der, stroke_width=4, fill_opacity=0.1,
                                 fill_color=color_der).shift(RIGHT * 4.5)
        pisos_der = VGroup(*[
            Line(edificio_der.get_left() + UP * y, edificio_der.get_right() + UP * y, color=color_der, stroke_width=3)
            for y in np.linspace(
                edificio_der.get_bottom()[1],
                edificio_der.get_top()[1],
                5  # 4 pisos → 5 puntos
            )[1:-1]
        ])

        # Edificio Central (Más alto)
        edificio_cen = Rectangle(width=3.5, height=5.5, color=color_cen, stroke_width=6, fill_opacity=0.15,
                                 fill_color=color_cen).shift(DOWN * 0.3)
        pisos_cen = VGroup(*[
            Line(edificio_cen.get_left() + UP * y, edificio_cen.get_right() + UP * y, color=color_cen, stroke_width=4)
            for y in np.linspace(-2.0, 2.0, 4)
        ])

        edificios = VGroup(edificio_izq, pisos_izq, edificio_der, pisos_der, edificio_cen, pisos_cen)

        # Animación de aparición
        self.play(LaggedStart(*[DrawBorderThenFill(mob) for mob in edificios], lag_ratio=0.1), run_time=6)
        self.wait(8)

        robot = Dot(point=edificio_cen.get_center() + DOWN * 0.5, color=YELLOW, radius=0.15)
        halo_robot = Circle(radius=0.4, color=YELLOW, fill_opacity=0.2, stroke_width=0).move_to(robot)

        self.play(
            FadeIn(halo_robot),
            GrowFromCenter(robot),
            run_time=2
        )
        self.wait(2)
        self.play(
            Flash(robot, color=YELLOW, flash_radius=0.6, line_length=0.3),
            run_time=1
        )

        antenas = VGroup()
        for edificio, color in [(edificio_izq, color_izq), (edificio_cen, color_cen), (edificio_der, color_der)]:
            for esquina in [UL, UR, DL, DR]:
                # Diseño de Antena
                punto = Dot(color=WHITE, radius=0.06)
                arco1 = Arc(radius=0.12, angle=PI, color=WHITE, stroke_width=2).next_to(punto, UP, buff=0.02)
                arco2 = Arc(radius=0.2, angle=PI, color=WHITE, stroke_width=2).next_to(punto, UP, buff=0.02)

                antena = VGroup(punto, arco1, arco2).scale(0.8)
                antena.move_to(edificio.get_corner(esquina) + 0.2 * np.sign(esquina))
                antenas.add(antena)

        self.play(LaggedStart(*[FadeIn(ant, scale=0.5) for ant in antenas], lag_ratio=1.5))
        self.wait(6)

        ondas_antenas = VGroup()
        for antena in antenas:
            # La onda nace desde el centro exacto de cada antena
            onda = Circle(radius=0.1, color=BLUE_B, stroke_width=2).move_to(antena.get_center())
            onda.generate_target()
            onda.target.scale(25).set_opacity(0)
            ondas_antenas.add(onda)

        # Hacemos que barran la pantalla suavemente
        self.play(LaggedStart(*[MoveToTarget(onda, rate_func=rate_functions.ease_out_sine) for onda in ondas_antenas],
                              lag_ratio=0.8), run_time=4)

        numeros_str = [
            ["-105", "-30", "-98"],
            ["-90", "-40", "-105"],
            ["-98", "-38", "-86"]
        ]
        matriz_completa = Matrix(numeros_str).set_color(GREEN).scale(0.8).to_edge(UP, buff=0.5).shift(RIGHT * 3)
        fondo_matriz = BackgroundRectangle(matriz_completa, color=BLACK, fill_opacity=0.8, buff=0.2)

        # Ocultamos los números (entradas)
        entradas = matriz_completa.get_entries()
        for entrada in entradas:
            entrada.set_opacity(0)

        brackets = matriz_completa.get_brackets()

        # Aparece el panel de la matriz vacío
        self.play(FadeIn(fondo_matriz), FadeIn(brackets, shift=DOWN))

        # Tenemos 12 antenas en total (0 a 11). Elegimos 9 distintas y esparcidas para la animación.
        # Índices: [Izquierda_Arriba, Centro_Abajo, Derecha_Arriba, Centro_Arriba...]
        indices_antenas = [0, 6, 9, 1, 4, 11, 2, 5, 8, 0]
        antenas_seleccionadas = [antenas[i] for i in indices_antenas]

        # Ráfaga de comunicación: Una línea por cada número en la matriz
        for antena, entrada_matriz in zip(antenas_seleccionadas, entradas):
            rayo = Line(robot.get_center(), antena.get_center(), color=GOLD, stroke_width=4)

            self.play(ShowPassingFlash(rayo.copy(), time_width=0.5), run_time=0.7)
            self.play(
                Flash(antena, color=GOLD, flash_radius=0.4),
                entrada_matriz.animate.set_opacity(1),
                Flash(entrada_matriz, color=GREEN, flash_radius=0.4),
                run_time=0.7
            )

        # Destello final
        grupo_matriz = VGroup(fondo_matriz, matriz_completa)
        self.play(Flash(grupo_matriz, color=GREEN, flash_radius=2.2, num_lines=15))
        self.wait(5)

        # Transición limpia al Acto 1
        self.play(FadeOut(Group(*self.mobjects)))

class Act1_Dimensionality(ThreeDScene):
    def construct(self):
        # Escena 1: 1D -> 2D -> 3D

        titulo = Text("La Maldición de la Dimensionalidad").to_edge(UP)

        self.add_fixed_in_frame_mobjects(titulo)
        self.add_sound("audio/proyecto4/Ac1_CURSE_P4.wav")
        self.play(Write(titulo))

        # 1D
        linea_1d = NumberLine(x_range=[-3, 3, 1], length=6).shift(DOWN)
        puntos_1d = VGroup(*[Dot(color=RED).move_to(linea_1d.n2p(i * 0.5)) for i in range(-2, 3)])
        self.play(Create(linea_1d), FadeIn(puntos_1d), run_time = 2.5)
        self.wait(1)

        # 2D
        ejes_2d = Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], x_length=6, y_length=6).shift(DOWN)
        puntos_2d = VGroup(
            *[Dot(color=RED).move_to(ejes_2d.c2p(i * 0.5 + random.uniform(-1, 1), i * 0.5 + random.uniform(-1, 1))) for
              i in range(-2, 3)])
        self.play(Transform(linea_1d, ejes_2d), Transform(puntos_1d, puntos_2d), run_time = 2.5)
        self.wait(1)

        # 3D (Cambiamos la perspectiva de la cámara)
        ejes_3d = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3])
        puntos_3d = VGroup(
            *[Dot3D(color=RED).move_to(ejes_3d.c2p(random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2)))
              for _ in range(5)])

        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, run_time=3)
        self.play(ReplacementTransform(linea_1d, ejes_3d), ReplacementTransform(puntos_1d, puntos_3d))
        self.wait(2)

        self.play(FadeOut(ejes_3d), FadeOut(puntos_3d), run_time = 2)

        # 1. Saturación: crear miles de puntos rojos dentro del cubo
        multitud = VGroup()
        for _ in range(500):  # 1500 es el punto dulce para que se vea lleno sin lagear el render
            punto = Dot3D(color=RED, radius=0.03).move_to([
                random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3)
            ])
            multitud.add(punto)

        self.play(LaggedStart(*[FadeIn(p, scale=0.1) for p in multitud], lag_ratio=0.001), run_time=6)

        # 2. Parpadeo de sobrecarga
        self.play(
            *[p.animate.set_color(RED_A).scale(1.5) for p in multitud[:150]],
            rate_func=rate_functions.there_and_back,
            run_time=1
        )

        # 3. Giro rápido de cámara
        self.begin_ambient_camera_rotation(rate=0.5)
        self.wait(2)

        # 4. Explosión en líneas de neón (simulando 520 dimensiones)
        lineas_neon = VGroup()
        for _ in range(150):
            direccion = np.array([random.uniform(-1, 1) for _ in range(3)])
            direccion = direccion / np.linalg.norm(direccion)  # Asegurar que estallen en círculo perfecto
            linea = Line(ORIGIN, direccion * 10, color=interpolate_color(BLUE, PURPLE, random.random()))
            linea.set_stroke(width=random.uniform(1, 4), opacity=0.8)
            lineas_neon.add(linea)

        # Los puntos se alejan y desaparecen mientras nacen las líneas
        self.play(
            LaggedStart(*[Create(line) for line in lineas_neon], lag_ratio=0.01),
            multitud.animate.shift(OUT * 15).set_opacity(0),
            run_time=2
        )
        self.stop_ambient_camera_rotation()
        self.wait(0.5)

        # IMPORTANTE: Resetear la cámara y limpiar para la gráfica 2D
        self.move_camera(phi=0, theta=-90 * DEGREES, run_time=1)

        # Borrar todo excepto el título que está pegado a la pantalla
        mobs_to_remove = [mob for mob in self.mobjects if mob != titulo]
        self.play(FadeOut(Group(*mobs_to_remove)))

        ejes_var = Axes(x_range=[0, 100, 20], y_range=[0, 1, 0.2], x_length=8, y_length=5)
        labels = ejes_var.get_axis_labels(Text("Componentes").scale(0.5), Text("Varianza").scale(0.5))

        # Curva dorada (simulando varianza acumulada logarítmica)
        curva = ejes_var.plot(lambda x: 1 - np.exp(-x / 25), color=GOLD)
        linea_90 = ejes_var.get_horizontal_line(ejes_var.c2p(100, 0.9), color=WHITE, line_func=DashedLine)
        texto_90 = Text("90%").scale(0.5).next_to(linea_90, LEFT)

        self.play(Create(ejes_var), Create(labels))
        self.wait(4)
        self.play(Create(curva), run_time=3)
        self.play(Create(linea_90), Write(texto_90))
        self.wait(2)

        self.play(FadeOut(Group(*self.mobjects)))  # Limpiamos la pantalla

        # Título principal arriba
        titulo_resultados = Text("Resultados de Reducción", font_size=40).to_edge(UP)
        self.play(Write(titulo_resultados), run_time = 0.5)

        # 1. Cargar las imágenes
        # Usamos .set_width(6) porque la pantalla de Manim mide 14 de ancho.
        # Así aseguramos que ocupen casi la mitad cada una sin tocarse.
        img_pca = ImageMobject("image/pca.png").set_width(6).to_edge(LEFT, buff=0.5).shift(DOWN * 0.5)
        img_tsne = ImageMobject("image/tsne.png").set_width(6).to_edge(RIGHT, buff=0.5).shift(DOWN * 0.5)

        # 2. Textos simples para identificar cada imagen
        txt_pca = Text("PCA (Lineal)", font_size=28).next_to(img_pca, UP)
        txt_tsne = Text("t-SNE (No Lineal)", font_size=28).next_to(img_tsne, UP)

        # 3. Mostrar todo de forma limpia
        self.play(
            FadeIn(img_pca), FadeIn(txt_pca),
            FadeIn(img_tsne), FadeIn(txt_tsne),
            run_time=1
        )
        self.wait(2)


from manim import *
import numpy as np
import random


class Act2_Clustering(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a0a"
        random.seed(42)
        self.add_sound("audio/proyecto4/Act2_cluster_p4.wav")
        titulo = Text("Aprendizaje No Supervisado", color=WHITE)
        self.play(Write(titulo))
        self.wait(7)
        self.remove(titulo)

        subtitulo_km = Text("1. K-Means Clustering", color=BLUE_C, font_size=36).to_edge(UP)
        self.play(FadeIn(subtitulo_km))

        # Nube de puntos grises
        puntos_km = VGroup(*[
            Dot([random.uniform(-4, 4), random.uniform(-2, 2), 0], color=DARK_GREY, radius=0.08)
            for _ in range(60)
        ])

        # Corrección: Usar LaggedStart para el efecto en cascada
        self.play(LaggedStart(*[FadeIn(p) for p in puntos_km], lag_ratio=0.01), run_time=1.5)

        # Estrellas (centroides) que caen del cielo
        c1 = Star(color=RED, fill_opacity=1).scale(0.3).move_to([-3, 4, 0])
        c2 = Star(color=GREEN, fill_opacity=1).scale(0.3).move_to([3, 4, 0])
        c3 = Star(color=YELLOW, fill_opacity=1).scale(0.3).move_to([0, 4, 0])
        centroides = VGroup(c1, c2, c3)

        # Caen con pequeña vibración
        self.play(
            c1.animate.move_to([-3, 2, 0]),
            c2.animate.move_to([3, 2, 0]),
            c3.animate.move_to([0, -2, 0]),
            run_time=3,
            rate_func=rate_functions.ease_out_bounce
        )
        self.play(
            Flash(c1, color=RED, flash_radius=0.4),
            Flash(c2, color=GREEN, flash_radius=0.4),
            Flash(c3, color=YELLOW, flash_radius=0.4)
        )

        # Atracción: los puntos se mueven ligeramente hacia su centroide más cercano
        asignaciones = []
        desplazamientos = []
        for p in puntos_km:
            d1 = np.linalg.norm(p.get_center() - c1.get_center())
            d2 = np.linalg.norm(p.get_center() - c2.get_center())
            d3 = np.linalg.norm(p.get_center() - c3.get_center())

            if d1 < d2 and d1 < d3:
                asignaciones.append(RED)
                vector = c1.get_center() - p.get_center()
                desplazamientos.append(p.get_center() + vector * 0.2)
            elif d2 < d1 and d2 < d3:
                asignaciones.append(GREEN)
                vector = c2.get_center() - p.get_center()
                desplazamientos.append(p.get_center() + vector * 0.2)
            else:
                asignaciones.append(YELLOW)
                vector = c3.get_center() - p.get_center()
                desplazamientos.append(p.get_center() + vector * 0.2)

        # Animación: cambio de color y pequeño desplazamiento simultáneos
        self.play(
            *[p.animate.set_color(asignaciones[i]).move_to(desplazamientos[i]) for i, p in enumerate(puntos_km)],
            run_time=1.5
        )

        # Recalcular centroides
        pts_rojos = [p.get_center() for i, p in enumerate(puntos_km) if asignaciones[i] == RED]
        pts_verdes = [p.get_center() for i, p in enumerate(puntos_km) if asignaciones[i] == GREEN]
        pts_amarillos = [p.get_center() for i, p in enumerate(puntos_km) if asignaciones[i] == YELLOW]

        new_c1 = np.mean(pts_rojos, axis=0) if pts_rojos else c1.get_center()
        new_c2 = np.mean(pts_verdes, axis=0) if pts_verdes else c2.get_center()
        new_c3 = np.mean(pts_amarillos, axis=0) if pts_amarillos else c3.get_center()

        # Mover centroides con efecto de "succión"
        self.play(
            c1.animate.move_to(new_c1),
            c2.animate.move_to(new_c2),
            c3.animate.move_to(new_c3),
            run_time=1.5
        )

        # Pequeña vibración al llegar
        self.play(
            c1.animate.shift(0.05 * UP), c2.animate.shift(0.05 * RIGHT), c3.animate.shift(0.05 * LEFT),
            run_time=0.1
        )
        self.play(
            c1.animate.shift(0.05 * DOWN), c2.animate.shift(0.05 * LEFT), c3.animate.shift(0.05 * RIGHT),
            run_time=0.1
        )
        self.wait(0.5)

        # Corrección: Círculos de convergencia seguros
        circulos = VGroup(
            Circle(radius=1.5, color=RED, stroke_width=2).move_to(c1),
            Circle(radius=1.5, color=GREEN, stroke_width=2).move_to(c2),
            Circle(radius=1.5, color=YELLOW, stroke_width=2).move_to(c3)
        )
        self.play(Create(circulos))
        self.wait(3)

        # Limpieza y transición a imagen real
        self.play(FadeOut(puntos_km), FadeOut(centroides), FadeOut(circulos))
        img_kmeans = ImageMobject("image/kmeans.png").set_width(7).shift(DOWN * 0.5)
        self.play(FadeIn(img_kmeans, shift=UP * 0.5))
        self.wait(2)
        self.play(FadeOut(img_kmeans), FadeOut(subtitulo_km))


        subtitulo_db = Text("2. DBSCAN (Densidad)", color=PURPLE_C, font_size=36).to_edge(UP)
        self.play(FadeIn(subtitulo_db))

        # Nube en forma de "C" y puntos de ruido aislados
        puntos_db = VGroup(
            *[Dot([np.cos(t) * 2.5 + random.uniform(-0.2, 0.2), np.sin(t) * 2.5 + random.uniform(-0.2, 0.2), 0],
                  color=DARK_GREY) for t in np.linspace(0.5, 5.5, 45)],
            *[Dot([random.uniform(-4, 4), random.uniform(-2, 2), 0], color=DARK_GREY) for _ in range(12)]
        )
        self.play(LaggedStart(*[FadeIn(p) for p in puntos_db], lag_ratio=0.01), run_time=1)

        # 1. El Core Point (Punto Semilla) inicia
        semilla = puntos_db[10]
        self.play(semilla.animate.set_color(PURPLE).scale(1.5))

        # 2. Primer Ping de Radar (Radio Epsilon)
        radio_eps = 1.2
        radar1 = Circle(radius=radio_eps, color=PURPLE, stroke_width=3).move_to(semilla)
        self.play(Create(radar1))
        self.play(radar1.animate.scale(1.5).set_opacity(0), run_time=1)

        # Se contagian los vecinos directos
        vecinos_directos = [p for p in puntos_db if
                            p != semilla and np.linalg.norm(p.get_center() - semilla.get_center()) < radio_eps]
        self.play(*[p.animate.set_color(PURPLE_B).scale(1.2) for p in vecinos_directos], run_time=0.8)

        # 3. Segundo Ping Masivo (Los vecinos buscan a sus propios vecinos)
        radares_masivos = VGroup(
            *[Circle(radius=radio_eps, color=PURPLE_B, stroke_width=2).move_to(v) for v in vecinos_directos])
        self.play(FadeIn(radares_masivos))
        self.play(radares_masivos.animate.scale(1.5).set_opacity(0), run_time=0.8)

        # Contagio de toda la cadena (El cluster completo)
        cluster_completo = [p for p in puntos_db if p not in vecinos_directos and p != semilla and np.linalg.norm(
            p.get_center()) < 3.0]  # Filtramos la "C" aprox
        self.play(*[p.animate.set_color(PURPLE_A) for p in cluster_completo], run_time=0.8)

        # 4. El Ruido se apaga
        ruido = [p for p in puntos_db if p.get_color() == DARK_GREY]
        self.play(*[p.animate.set_color(GRAY_D).scale(0.5) for p in ruido], run_time=0.5)
        self.wait(1)

        # Transición a la imagen real
        self.play(FadeOut(puntos_db))
        img_dbscan = ImageMobject("image/dbscan.png").set_width(7).shift(DOWN * 0.5)
        self.play(FadeIn(img_dbscan, shift=UP * 0.5))
        self.wait(2)
        self.play(FadeOut(img_dbscan), FadeOut(subtitulo_db))

        subtitulo_agg = Text("3. Agglomerative Clustering", color=TEAL_C, font_size=36).to_edge(UP)
        self.play(FadeIn(subtitulo_agg))

        # Colocamos 6 puntos alineados abajo para construir el árbol claramente
        puntos_agg = VGroup(
            Dot([-3, -2, 0], color=TEAL),
            Dot([-2, -2, 0], color=TEAL),
            Dot([-0.5, -2, 0], color=TEAL),
            Dot([0.5, -2, 0], color=TEAL),
            Dot([2, -2, 0], color=TEAL),
            Dot([3, -2, 0], color=TEAL)
        )
        self.play(LaggedStart(*[GrowFromCenter(p) for p in puntos_agg], lag_ratio=0.1), run_time=0.4)

        # Función auxiliar para dibujar un "puente" de agrupación (jerarquía)
        def dibujar_puente(p1_pos, p2_pos, altura, color_puente):
            linea_izq = Line(p1_pos, [p1_pos[0], altura, 0], color=color_puente)
            linea_der = Line(p2_pos, [p2_pos[0], altura, 0], color=color_puente)
            linea_hor = Line([p1_pos[0], altura, 0], [p2_pos[0], altura, 0], color=color_puente)
            centro_superior = [(p1_pos[0] + p2_pos[0]) / 2, altura, 0]
            return VGroup(linea_izq, linea_der, linea_hor), centro_superior

        # Paso 1: Agrupamos los pares más cercanos (Nivel inferior)
        puente1, nodo1 = dibujar_puente(puntos_agg[0].get_center(), puntos_agg[1].get_center(), -1, TEAL_D)
        puente2, nodo2 = dibujar_puente(puntos_agg[2].get_center(), puntos_agg[3].get_center(), -1, TEAL_D)
        puente3, nodo3 = dibujar_puente(puntos_agg[4].get_center(), puntos_agg[5].get_center(), -1, TEAL_D)

        self.play(Create(puente1), Create(puente2), Create(puente3), run_time=0.7)


        puente4, nodo4 = dibujar_puente(nodo2, nodo3, 0.5, TEAL_C)
        self.play(Create(puente4), run_time=0.7)

        puente_raiz, nodo_raiz = dibujar_puente(nodo1, nodo4, 2, TEAL_B)
        self.play(Create(puente_raiz), run_time=0.7)
        self.wait(1)


        todo_el_dendrograma = VGroup(puntos_agg, puente1, puente2, puente3, puente4, puente_raiz)
        self.play(FadeOut(todo_el_dendrograma))

        img_agg = ImageMobject("image/agg.png").set_width(7).shift(DOWN * 0.5)
        self.play(FadeIn(img_agg, shift=UP * 0.5))
        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)))

class Act3_Metrics(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a0a"
        self.add_sound("audio/proyecto4/Ac3_metrica_p4.wav")

        titulo_sil = Text("Silhouette Score").to_edge(UP)
        self.play(Write(titulo_sil))

        # SUBIMOS LOS PUNTOS: Le añadimos un .shift(UP * 1) a todo el grupo
        puntos = VGroup(
            Dot(color=WHITE).shift(LEFT * 2),
            Dot(color=GREEN).shift(LEFT * 1),
            Dot(color=GREEN).shift(LEFT * 1.5 + DOWN * 0.5),
            Dot(color=GREEN).shift(LEFT * 2.5 + UP * 0.3),
            Dot(color=BLUE).shift(RIGHT * 2),
            Dot(color=BLUE).shift(RIGHT * 1 + DOWN * 0.8),
            Dot(color=BLUE).shift(RIGHT * 2.5 + UP * 0.2),
        ).shift(UP * 1)

        etiqueta_punto = Text("Punto i", font_size=20).next_to(puntos[0], DOWN)

        grupo_a = VGroup(puntos[1], puntos[2], puntos[3])
        grupo_b = VGroup(puntos[4], puntos[5], puntos[6])

        circulo_a = Ellipse(width=2.5, height=1.5, color=GREEN, stroke_width=2).move_to(grupo_a)
        circulo_b = Ellipse(width=2.5, height=1.5, color=BLUE, stroke_width=2).move_to(grupo_b)

        self.play(FadeIn(puntos), Write(etiqueta_punto))
        self.play(Create(circulo_a), Create(circulo_b))

        flecha_a = Arrow(puntos[0].get_center(), grupo_a.get_center(), color=GREEN, buff=0.2)
        flecha_b = Arrow(puntos[0].get_center(), grupo_b.get_center(), color=BLUE, buff=0.2)

        etiq_a = Text("a (cohesión)", color=GREEN, font_size=20).next_to(flecha_a, UP)
        etiq_b = Text("b (separación)", color=BLUE, font_size=20).next_to(flecha_b, RIGHT)

        self.play(GrowArrow(flecha_a), Write(etiq_a))
        self.play(GrowArrow(flecha_b), Write(etiq_b))
        self.wait(1)

        # SUBIMOS LA FÓRMULA: Cambiamos de DOWN * 2 a DOWN * 0.5
        formula_sil = MathTex(r"S = \frac{b - a}{\max(a, b)}").scale(1.5)
        formula_sil.move_to(DOWN * 0.5)

        leyenda_a = Text("a = Cohesión (Distancia interna)", color=GREEN).scale(0.5).next_to(formula_sil, DOWN,
                                                                                             buff=0.5)
        leyenda_b = Text("b = Separación (Distancia al vecino)", color=BLUE).scale(0.5).next_to(leyenda_a, DOWN,
                                                                                                buff=0.2)

        self.play(Write(formula_sil), FadeIn(leyenda_a, shift=UP), FadeIn(leyenda_b, shift=UP))
        self.wait(9)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.5)


        titulo_ari = Text("Adjusted Rand Index (ARI)").to_edge(UP)
        self.play(Write(titulo_ari))

        # SUBIMOS LOS PUNTOS: Cambiamos shift(UP * 0.5) a shift(UP * 1.5)
        pts = VGroup(*[Dot(color=GRAY) for _ in range(8)])
        pts.arrange(RIGHT, buff=0.8).shift(UP * 1.5)

        etiquetas_reales = VGroup(
            Text("A", color=BLUE, font_size=20).next_to(pts[0], UP),
            Text("A", color=BLUE, font_size=20).next_to(pts[1], UP),
            Text("B", color=GREEN, font_size=20).next_to(pts[2], UP),
            Text("B", color=GREEN, font_size=20).next_to(pts[3], UP),
            Text("C", color=YELLOW, font_size=20).next_to(pts[4], UP),
            Text("C", color=YELLOW, font_size=20).next_to(pts[5], UP),
            Text("D", color=PURPLE, font_size=20).next_to(pts[6], UP),
            Text("D", color=PURPLE, font_size=20).next_to(pts[7], UP),
        )
        self.play(FadeIn(pts), Write(etiquetas_reales))

        etiquetas_pred = VGroup(
            Text("A", color=BLUE, font_size=20).next_to(pts[0], DOWN),
            Text("A", color=BLUE, font_size=20).next_to(pts[1], DOWN),
            Text("B", color=GREEN, font_size=20).next_to(pts[2], DOWN),
            Text("C", color=YELLOW, font_size=20).next_to(pts[3], DOWN),
            Text("C", color=YELLOW, font_size=20).next_to(pts[4], DOWN),
            Text("D", color=PURPLE, font_size=20).next_to(pts[5], DOWN),
            Text("D", color=PURPLE, font_size=20).next_to(pts[6], DOWN),
            Text("D", color=PURPLE, font_size=20).next_to(pts[7], DOWN),
        )
        self.play(Write(etiquetas_pred))

        aciertos = VGroup(pts[0], pts[1], pts[2], pts[4], pts[6], pts[7])
        errores = VGroup(pts[3], pts[5])
        self.play(
            *[p.animate.set_color(GREEN) for p in aciertos],
            *[p.animate.set_color(RED) for p in errores],
            run_time=2
        )

        # SUBIMOS LA FÓRMULA: Cambiamos DOWN * 1.5 a DOWN * 0.2
        formula_ari = MathTex(r"ARI = \frac{\text{Aciertos Reales} - \text{Azar}}{\text{Máximos} - \text{Azar}}").scale(
            1.2)
        formula_ari.move_to(DOWN * 0.2)

        leyenda_ari1 = Text("Aciertos Reales: Agrupados en el piso correcto", color=GREEN).scale(0.5).next_to(
            formula_ari, DOWN, buff=0.5)
        leyenda_ari2 = Text("Azar: Lo que acertaría un modelo aleatorio", color=GRAY).scale(0.5).next_to(leyenda_ari1,
                                                                                                           DOWN,
                                                                                                           buff=0.2)
        leyenda_ari3 = Text("Máximos: Perfección absoluta (1.0)", color=GOLD).scale(0.5).next_to(leyenda_ari2, DOWN,
                                                                                                   buff=0.2)

        self.play(Write(formula_ari), FadeIn(leyenda_ari1), FadeIn(leyenda_ari2), FadeIn(leyenda_ari3))
        self.wait(7)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.5)

        titulo_final = Text("Expectativa vs Realidad").to_edge(UP)
        self.play(FadeIn(titulo_final))

        # Corrección: Ejes y barras bien posicionadas sin usar c2p inestable
        ejes_barras = Axes(x_range=[0, 3, 1], y_range=[0, 1, 0.2], x_length=5, y_length=4).shift(DOWN * 0.5)

        barra_sil = Rectangle(width=1, height=3, color=GREEN, fill_opacity=0.8).move_to(ejes_barras.c2p(1, 0),
                                                                                        aligned_edge=DOWN)
        barra_ari = Rectangle(width=1, height=0.4, color=RED, fill_opacity=0.8).move_to(ejes_barras.c2p(2, 0),
                                                                                        aligned_edge=DOWN)

        label_sil = Text("Silhouette (Alto)", color=GREEN).scale(0.5).next_to(barra_sil, DOWN, buff=0.3)
        label_ari = Text("ARI (Bajo)", color=RED).scale(0.5).next_to(barra_ari, DOWN, buff=0.3)

        self.play(Create(ejes_barras), FadeIn(label_sil), FadeIn(label_ari))
        self.play(GrowFromEdge(barra_sil, DOWN), run_time=1.5)
        self.play(GrowFromEdge(barra_ari, DOWN), run_time=1.5)

        # Grieta
        grieta = Line(barra_ari.get_top(), barra_ari.get_bottom(), color=WHITE, stroke_width=3)
        grieta.rotate(0.2)
        self.play(Create(grieta))

        # Efecto Terremoto
        ruido = VGroup(ejes_barras, barra_sil, barra_ari, label_sil, label_ari, grieta)
        for _ in range(6):
            self.play(ruido.animate.shift(LEFT * 0.1), run_time=0.05)
            self.play(ruido.animate.shift(RIGHT * 0.2), run_time=0.05)
            self.play(ruido.animate.shift(LEFT * 0.1), run_time=0.05)

        conclusion = Text("El clustering falló al separar los pisos (eje Z).", color=YELLOW, font_size=28).shift(UP * 2)
        self.play(Write(conclusion))
        self.wait(1)


        self.play(FadeOut(Group(*self.mobjects)))

        titulo_resultados = Text("Resultados (Matplotlib)", color=WHITE).to_edge(UP)
        self.play(Write(titulo_resultados), run_time=0.4)

        # Cargar las 3 imágenes (K-Means, DBSCAN, Agglomerative) de resultados
        # Se asume que tienes gráficas de barras con los scores reales guardadas.
        img_res_km = ImageMobject("image/res_kmeans.png")
        img_res_db = ImageMobject("image/res_dbscan.png")
        img_res_agg = ImageMobject("image/res_agg.png")

        # Las forzamos a tener el mismo alto para que el Group.arrange() las ponga bonitas
        for img in [img_res_km, img_res_db, img_res_agg]:
            img.height = 3.5

        grupo_resultados = Group(img_res_km, img_res_db, img_res_agg).arrange(RIGHT, buff=0.5).shift(DOWN * 0.5)

        self.play(FadeIn(grupo_resultados, shift=UP), run_time=1.5)
        self.wait(5)

        # Transición limpia para el Acto 4
        self.play(FadeOut(Group(*self.mobjects)))


class Act4_DeepLearning(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a0f"
        self.add_sound("audio/proyecto4/Ac4_1_Auto_p4.wav")
        titulo = Text("Deep Learning: Autoencoder + Clasificador", color=BLUE_C, font_size=36).to_edge(UP)
        self.play(Write(titulo))


        def crear_columna_neuronas(n_visual, x_pos, color=WHITE, radio=0.08, espaciado=0.22):
            neuronas = VGroup()
            for i in range(n_visual):
                y_pos = (i - n_visual / 2 + 0.5) * espaciado
                dot = Dot([x_pos, y_pos, 0], radius=radio, color=color)
                neuronas.add(dot)
            return neuronas

        def conectar_capas(capa1, capa2, color_linea=WHITE, opacidad=0.15):
            """Crea todas las líneas (pesos) entre dos capas."""
            conexiones = VGroup()
            for n1 in capa1:
                for n2 in capa2:
                    linea = Line(n1.get_center(), n2.get_center(), color=color_linea, stroke_width=1,
                                 stroke_opacity=opacidad)
                    conexiones.add(linea)
            return conexiones

        x_in, x_h1, x_bot, x_h2, x_out = -5.5, -2.75, 0, 2.75, 5.5

        capa_in = crear_columna_neuronas(25, x_in, color=BLUE_E)
        capa_h1 = crear_columna_neuronas(12, x_h1, color=BLUE_C)
        capa_bot = crear_columna_neuronas(6, x_bot, color=YELLOW, radio=0.12, espaciado=0.4)
        capa_h2 = crear_columna_neuronas(12, x_h2, color=BLUE_C)
        capa_out = crear_columna_neuronas(25, x_out, color=BLUE_E)

        # Crear cables
        con_in_h1 = conectar_capas(capa_in, capa_h1, BLUE_E)
        con_h1_bot = conectar_capas(capa_h1, capa_bot, BLUE_C)
        con_bot_h2 = conectar_capas(capa_bot, capa_h2, YELLOW)
        con_h2_out = conectar_capas(capa_h2, capa_out, BLUE_C)

        todas_las_conexiones = VGroup(con_in_h1, con_h1_bot, con_bot_h2, con_h2_out)
        autoencoder = VGroup(capa_in, capa_h1, capa_bot, capa_h2, capa_out)

        # Aparecen primero los nodos y luego los cables
        self.play(LaggedStart(*[FadeIn(capa) for capa in autoencoder], lag_ratio=0.1), run_time=3)
        self.play(FadeIn(todas_las_conexiones), run_time=3)

        etiq_in = Text("520", color=BLUE_E, font_size=20).next_to(capa_in, DOWN, buff=0.3)
        etiq_h1 = Text("128", color=BLUE_C, font_size=20).next_to(capa_h1, DOWN, buff=0.3)
        etiq_bn = Text("32", color=YELLOW, font_size=24, weight=BOLD).next_to(capa_bot, DOWN, buff=0.3)
        etiq_h2 = Text("128", color=BLUE_C, font_size=20).next_to(capa_h2, DOWN, buff=0.3)
        etiq_out = Text("520", color=BLUE_E, font_size=20).next_to(capa_out, DOWN, buff=0.3)

        etiquetas_ae = VGroup(etiq_in, etiq_h1, etiq_bn, etiq_h2, etiq_out)
        self.play(FadeIn(etiquetas_ae, shift=UP * 0.2))
        for i in range(0, 5):
            # --- ANIMACIÓN: FORWARD (Luz de ida) ---
            self.wait(0.4)
            flujo_ida = WHITE

            for capa_origen, conexion, capa_destino in zip(
                    [capa_in, capa_h1, capa_bot, capa_h2],
                    [con_in_h1, con_h1_bot, con_bot_h2, con_h2_out],
                    [capa_h1, capa_bot, capa_h2, capa_out]
            ):
                self.play(
                    ShowPassingFlash(conexion.copy().set_color(flujo_ida).set_stroke(opacity=1.0, width=2.5),
                                     time_width=1),
                    run_time=0.8
                )


            # --- ANIMACIÓN: BACKPROPAGATION (Luz de vuelta) ---
            flujo_vuelta = ORANGE
            for capa_origen, conexion, capa_destino in zip(
                    [capa_out, capa_h2, capa_bot, capa_h1],
                    [con_h2_out, con_bot_h2, con_h1_bot, con_in_h1],
                    [capa_h2, capa_bot, capa_h1, capa_in]
            ):


                # TRUCO PRO: Invertimos el inicio y fin de CADA cable individualmente
                conexion_inversa = VGroup(*[linea.copy().reverse_direction() for linea in conexion])

                # 2. La luz viaja de DERECHA a IZQUIERDA
                self.play(
                    ShowPassingFlash(
                        conexion_inversa.set_color(flujo_vuelta).set_stroke(opacity=1.0, width=2.5),
                        time_width=1
                    ),
                    run_time=0.8
                )


        self.play(Flash(capa_bot, color=YELLOW, flash_radius=1.5, num_lines=12))
        self.wait(5)


        mobs_a_borrar = VGroup(
            capa_in, capa_h1, capa_h2, capa_out,
            con_in_h1, con_h1_bot, con_bot_h2, con_h2_out,
            etiq_in, etiq_h1, etiq_h2, etiq_out
        )
        self.add_sound("audio/proyecto4/Ac4_2_MLP_p4.wav")
        self.play(mobs_a_borrar.animate.set_opacity(0).shift(DOWN * 0.5), run_time=3)
        self.remove(*mobs_a_borrar)

        # Movemos el Bottleneck bien a la izquierda para que todo quepa después
        bottleneck_group = VGroup(capa_bot, etiq_bn)
        self.play(bottleneck_group.animate.move_to([-5.5, 0, 0]), run_time=2)

        # Coordenadas X fijas y calculadas para que no se salga de la pantalla
        x_bot_nuevo = -5.5
        x_mlp1 = -2.5
        x_mlp2 = 0.5
        x_salida = 3.5

        mlp_h1 = crear_columna_neuronas(10, x_mlp1, color=TEAL, radio=0.1)
        mlp_h2 = crear_columna_neuronas(6, x_mlp2, color=TEAL_C, radio=0.1, espaciado=0.3)

        salida_neurons = VGroup()
        etiquetas_salida = VGroup()

        edificios_pisos = [
            "Edif 0, P1", "Edif 0, P2", "Edif 0, P3", "Edif 0, P4",
            "Edif 1, P1", "Edif 1, P2", "Edif 1, P3", "Edif 1, P4",
            "Edif 2, P1", "Edif 2, P2", "Edif 2, P3", "Edif 2, P4", "Edif 2, P5"
        ]
        colores_salida = [BLUE_C] * 4 + [TEAL_C] * 4 + [PURPLE_C] * 5

        for idx, (texto, color) in enumerate(zip(edificios_pisos, colores_salida)):
            y_pos = (idx - 6) * 0.45
            neurona = Dot([x_salida, y_pos, 0], radius=0.12, color=color)
            etiq = Text(texto, font_size=16, color=color).next_to(neurona, RIGHT, buff=0.2)
            salida_neurons.add(neurona)
            etiquetas_salida.add(etiq)

        # Conexiones del MLP
        con_bot_mlp1 = conectar_capas(capa_bot, mlp_h1, TEAL, opacidad=0.15)
        con_mlp1_mlp2 = conectar_capas(mlp_h1, mlp_h2, TEAL_C, opacidad=0.15)
        con_mlp2_salida = conectar_capas(mlp_h2, salida_neurons, PURPLE_A, opacidad=0.1)

        conexiones_mlp = VGroup(con_bot_mlp1, con_mlp1_mlp2, con_mlp2_salida)

        self.play(
            LaggedStart(
                FadeIn(mlp_h1), FadeIn(mlp_h2), FadeIn(salida_neurons), Write(etiquetas_salida),
                lag_ratio=0.1
            ), run_time=3
        )
        self.play(FadeIn(conexiones_mlp), run_time=2.5)

        # Etiquetas del MLP
        etiq_mlp1 = Text("64", color=TEAL, font_size=20).next_to(mlp_h1, DOWN, buff=0.3)
        etiq_mlp2 = Text("32", color=TEAL_C, font_size=20).next_to(mlp_h2, DOWN, buff=0.3)
        etiq_salida = Text("13 Zonas", color=WHITE, font_size=20, weight=BOLD).next_to(salida_neurons, DOWN, buff=0.3)
        self.play(FadeIn(VGroup(etiq_mlp1, etiq_mlp2, etiq_salida)))

        # ---------------------------------------------------------
        # 4. ENTRENAMIENTO DEL MLP (Épocas) Y PREDICCIÓN FINAL
        # ---------------------------------------------------------
        # Bucle de 3 "Épocas" de entrenamiento (Ida y Vuelta)
        for i in range(5):
            # --- ANIMACIÓN: FORWARD (Luz de ida) ---
            self.wait(0.4)
            flujo_ida = WHITE

            for conexion in [con_bot_mlp1, con_mlp1_mlp2, con_mlp2_salida]:
                self.play(
                    ShowPassingFlash(
                        conexion.copy().set_color(flujo_ida).set_stroke(opacity=1.0, width=2.5),
                        time_width=1
                    ),
                    run_time=0.8
                )

            # --- ANIMACIÓN: BACKPROPAGATION (Luz de vuelta) ---
            flujo_vuelta = ORANGE
            for conexion in [con_mlp2_salida, con_mlp1_mlp2, con_bot_mlp1]:
                # TRUCO PRO: Invertimos el inicio y fin de CADA cable individualmente
                conexion_inversa = VGroup(*[linea.copy().reverse_direction() for linea in conexion])

                # La luz viaja de DERECHA a IZQUIERDA
                self.play(
                    ShowPassingFlash(
                        conexion_inversa.set_color(flujo_vuelta).set_stroke(opacity=1.0, width=2.5),
                        time_width=1
                    ),
                    run_time=0.8
                )

        # Hacemos un último Forward (solo de ida) en color verde para mostrar la decisión final
        flujo_decision = GREEN
        self.wait(0.5)
        for conexion in [con_bot_mlp1, con_mlp1_mlp2, con_mlp2_salida]:
            self.play(
                ShowPassingFlash(
                    conexion.copy().set_color(flujo_decision).set_stroke(opacity=1.0, width=3.0),
                    time_width=0.5
                ),
                run_time=0.4
            )

        # Índice 11 es "Edif 2, P4"
        idx_ganador = 11
        neurona_ganadora = salida_neurons[idx_ganador]
        etiq_ganadora = etiquetas_salida[idx_ganador]

        # Atenuamos incorrectas
        apagadas = VGroup(
            *[n for i, n in enumerate(salida_neurons) if i != idx_ganador],
            *[e for i, e in enumerate(etiquetas_salida) if i != idx_ganador]
        )
        self.play(apagadas.animate.set_opacity(0.15), run_time=0.5)

        # Resaltamos la correcta (Edif 2, P4)
        self.play(
            neurona_ganadora.animate.scale(2.5).set_color(GREEN),
            run_time=0.5
        )
        self.play(
            etiq_ganadora.animate.scale(1.5).set_color(GREEN).next_to(neurona_ganadora, RIGHT, buff=0.3),
            Flash(neurona_ganadora, color=GREEN, flash_radius=0.6),
            run_time=0.5
        )

        # Cartel final
        cartel = Text("98.5%", color=GREEN, font_size=24, weight=BOLD).next_to(etiq_ganadora, RIGHT, buff=0.5)
        marco_ganador = SurroundingRectangle(VGroup(neurona_ganadora, etiq_ganadora, cartel), color=GREEN,
                                             buff=0.10)

        self.play(Write(cartel), Create(marco_ganador))
        self.play(Flash(marco_ganador, color=GREEN, line_length=0.2, num_lines=20))
        self.wait(2)

        self.play(FadeOut(Group(*self.mobjects)))

class ConclusionScene(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a0a"
        self.add_sound("audio/proyecto4/Conclusion_p4.wav")

        titulo_matriz = Text("Matriz de Confusión (Red Neuronal)", color=WHITE).to_edge(UP).scale(0.8)
        self.play(Write(titulo_matriz), run_time = 2)

        # Creamos una cuadrícula 13x13 visual
        cuadricula = VGroup()
        for i in range(13):
            fila = VGroup()
            for j in range(13):
                cuadro = Square(side_length=0.35, fill_opacity=0.1, color=BLUE_E)
                cuadro.move_to(RIGHT * (j * 0.35) + DOWN * (i * 0.35))
                fila.add(cuadro)
            cuadricula.add(fila)

        cuadricula.center().shift(DOWN * 0.5)
        self.play(FadeIn(cuadricula), run_time=2)


        diagonal = VGroup(*[cuadricula[i][i] for i in range(13)])
        self.play(
            diagonal.animate.set_fill(TEAL, opacity=0.9).set_color(TEAL),
            run_time=4.0,
            rate_func=rate_functions.ease_in_out_sine
        )
        self.wait(2)

        # Cargar la imagen real y ajustar su tamaño (height=6 encaja perfecto en la pantalla)
        img_matrx = ImageMobject("image/matriz_real.png").set_height(6).shift(DOWN * 0.5)

        # Transición cruzada: se va la animación, entra tu gráfica de Python
        self.play(
            FadeOut(cuadricula),
            FadeIn(img_matrx),
            run_time=3
        )
        self.wait(5)

        # Limpieza para pasar a la siguiente escena
        self.play(FadeOut(titulo_matriz), FadeOut(img_matrx))

        # =========================================================
        # ESCENA 2: Pantalla Dividida (Comparativa)
        # =========================================================
        texto_vs = Text("VS").scale(1.5).move_to(ORIGIN)
        izq_texto = Text("Clustering Clásico\n(Precisión Baja)", color=RED).scale(0.6).move_to(LEFT * 4)
        der_texto = Text("Deep Learning (MLP)\n(Precisión Alta)", color=GREEN).scale(0.6).move_to(RIGHT * 4)

        self.play(FadeIn(izq_texto), FadeIn(texto_vs), FadeIn(der_texto))
        self.wait(5)
        self.play(FadeOut(Group(*self.mobjects)))

        color_izq = "#00ffff"
        color_der = "#00ffff"
        color_cen = "#ff66cc"

        # Edificio Izquierdo
        edificio_izq = Rectangle(width=2.2, height=4, color=color_izq, stroke_width=4, fill_opacity=0.1,
                                 fill_color=color_izq).shift(LEFT * 4.5)
        pisos_izq = VGroup(*[
            Line(edificio_izq.get_left() + UP * y, edificio_izq.get_right() + UP * y, color=color_izq, stroke_width=3)
            for y in np.linspace(edificio_izq.get_bottom()[1], edificio_izq.get_top()[1], 5)[1:-1]
        ])

        # Edificio Derecho
        edificio_der = Rectangle(width=2.2, height=4, color=color_der, stroke_width=4, fill_opacity=0.1,
                                 fill_color=color_der).shift(RIGHT * 4.5)
        pisos_der = VGroup(*[
            Line(edificio_der.get_left() + UP * y, edificio_der.get_right() + UP * y, color=color_der, stroke_width=3)
            for y in np.linspace(edificio_der.get_bottom()[1], edificio_der.get_top()[1], 5)[1:-1]
        ])

        # Edificio Central (5 pisos)
        edificio_cen = Rectangle(width=3.5, height=5.5, color=color_cen, stroke_width=6, fill_opacity=0.15,
                                 fill_color=color_cen).shift(DOWN * 0.3)
        pisos_cen = VGroup(*[
            Line(edificio_cen.get_left() + UP * y, edificio_cen.get_right() + UP * y, color=color_cen, stroke_width=4)
            for y in np.linspace(-2.0, 2.0, 4)
        ])

        edificios = VGroup(edificio_izq, pisos_izq, edificio_der, pisos_der, edificio_cen, pisos_cen)
        self.play(FadeIn(edificios), run_time = 3)

        # Re-crear antenas
        antenas = VGroup()
        for edificio, color in [(edificio_izq, color_izq), (edificio_cen, color_cen), (edificio_der, color_der)]:
            for esquina in [UL, UR, DL, DR]:
                punto = Dot(color=WHITE, radius=0.06)
                arco1 = Arc(radius=0.12, angle=PI, color=WHITE, stroke_width=2).next_to(punto, UP, buff=0.02)
                arco2 = Arc(radius=0.2, angle=PI, color=WHITE, stroke_width=2).next_to(punto, UP, buff=0.02)
                antena = VGroup(punto, arco1, arco2).scale(0.8)
                antena.move_to(edificio.get_corner(esquina) + 0.2 * np.sign(esquina))
                antenas.add(antena)

        self.play(FadeIn(antenas), run_time = 3)

        # El robot empieza en el Edificio Izquierdo, Piso 1
        pos_inicio = edificio_izq.get_bottom() + UP * 0.5
        robot = Dot(point=pos_inicio, color=GREEN, radius=0.15)
        halo = Circle(radius=0.4, color=GREEN, fill_opacity=0.2, stroke_width=0)

        # Hacemos que el halo siga al robot automáticamente
        halo.add_updater(lambda m: m.move_to(robot.get_center()))

        self.play(FadeIn(halo), GrowFromCenter(robot))
        self.play(Flash(robot, color=GREEN, flash_radius=1.2))

        # 1. Sube un poco en el Edificio Izquierdo
        self.play(robot.animate.shift(UP * 1.5), run_time=1.2)

        # 2. Salta al Edificio Derecho (Rastreo fallido) usando un arco
        pos_der = edificio_der.get_bottom() + UP * 1.5
        self.play(
            robot.animate.move_to(pos_der),
            path_arc=-1.5,  # Hace una curva en el aire
            run_time=1.2
        )
        self.play(robot.animate.shift(UP * 1.0), run_time=1)

        # 3. Salta al Edificio Central y llega exactamente al Piso 4 (La Predicción)
        pos_final = edificio_cen.get_center() + UP * 1.0  # Altura del Piso 4 aprox.
        self.play(
            robot.animate.move_to(pos_final),
            path_arc=1.2,
            run_time=1.2
        )

        # Bloqueo de Target: ¡Encontrado!
        self.play(
            robot.animate.scale(1.5),
            Flash(robot, color=GREEN, flash_radius=1.5, line_length=0.5, num_lines=16),
            run_time=0.8
        )

        # Cartel final épico
        cartel_final = Text("Localización 3D Completada", color=WHITE, weight=BOLD).scale(0.9).to_edge(UP)
        marco_cartel = SurroundingRectangle(cartel_final, color=GREEN, buff=0.2, stroke_width=2)

        self.play(Write(cartel_final), Create(marco_cartel))

        # Efecto de ondas confirmando la posición desde el robot
        ondas_finales = VGroup(
            *[Circle(radius=0.1, color=GREEN, stroke_width=2).move_to(robot) for _ in range(3)])
        self.play(
            LaggedStart(
                *[onda.animate.scale(30).set_opacity(0) for onda in ondas_finales],
                lag_ratio=0.3
            ),
            run_time=2
        )

        self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)))

        # Texto de "Gracias" con un gradiente de colores vibrantes
        texto_gracias = Text(
            "¡Gracias!",
            font_size=80,
            gradient=(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)
        )

        # Tu nombre debajo, en blanco limpio para contrastar
        texto_nombre = Text(
            "Carlos Angel Oriundo",
            font_size=40,
            color=WHITE
        ).next_to(texto_gracias, DOWN, buff=0.8)

        # Agrupamos para asegurar que el conjunto esté perfectamente centrado
        creditos = VGroup(texto_gracias, texto_nombre).move_to(ORIGIN)

        # Animación de aparición
        self.play(Write(texto_gracias), run_time=1.5)
        self.play(FadeIn(texto_nombre, shift=UP * 0.5), run_time=1)

        # Esperamos unos segundos para que se lea bien antes de terminar el video
        self.wait(4)

        # Fundido a negro final
        self.play(FadeOut(creditos))