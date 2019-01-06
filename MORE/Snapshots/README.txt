### Testdurchläufe 2019-01-04 ###

Polynomial Policy
 - Ohne training states:
    - Pol_4000iter.png: sample(1000, 20, 300, b, Q)
 - Mit training states:
    - Durch zufall wurde ein maximaler/konstanter Reward gefunden und MORE
      ist in einer Dauerschleife geendet:
       - ts_reward=1_beg.png
       - ts_reward=1_end.png
NN Policy:
 - Habe sample(10000, 20, 300, b, Q) aufgerufen und die NN Policy laufen lassen.
   Nach 3h ist sie mit folgenden Fehlermeldungen abgebrochen.
   NN_abort3.png zeigt aber, dass ein lokales Optimum irgendwie gefunden wurde,
   da die Varianz geringer wird.
    - NN_abort1.png:    Fehlermeldung
    - NN_abort2.png:    Fehlermeldung
    - NN_abort3.png:    np.diag(Q).sum.() wurde geringer



### Testdurchläufe 2019-01-05 ###

Polynomial Policy:
 - Mit training states:
    - Pol_ts_T1.png:    training_sample(.) mit number_of_thetas = 100
    - Pol_ts_T2.png:    training_sample(.) mit number_of_thetas = 20
 - Ohne Training states
    - Um der Problematik "Konstanter Reward" aus dem Weg zu gehen, habe ich den
      reward immer mit 1e5 multipliziert (siehe sample.py ca. line 62)
    - Das war das Ergebnis mit sample(1000, 20, 300, b, Q):
        - Pol_T1.png
        - Pol_T2.png
    - Da ich der Meinug war, dass es Sinnvoller sei mehr Theta Sample zu haben 
      (number_of_thetas) und weniger sample pro Theta, habe ich mit folgender
      konfiguration und geändertem Reward (r * 1e5) die Ergebnisse erhalten:
      -> NB: So läuft es viel schneller durch pro Iteration
      -> sample(10, 100, 300, b, Q)
        - Pol_T3.png
    - Mit sample(10, 100, 300, b, Q) (weil's schneller läuft) und unveränderten
      Reward (also r nicht mit 1e5 multipliziert) - nach 2000 Iterationen habe
      ich einen Schnappschuss gemacht. Ist so gut wie identisch mit 4000iter.png
      aus 2019_01_04
        - Pol_T4.png
        

