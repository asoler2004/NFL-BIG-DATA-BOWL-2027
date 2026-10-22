import psycopg2
from rdflib import Graph, Literal, Namespace, RDF, XSD, URIRef

# Define Namespaces
EX = Namespace("http://example.org/nfl/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")

g = Graph()
g.bind("ex", EX)
g.bind("foaf", FOAF)

# Connect to PostgreSQL
conn = psycopg2.connect("dbname=NFL user=postgres password=postgres.2026 host=localhost")
cur = conn.cursor()

# 1. Map Players
cur.execute("SELECT nfl_id, display_name, draft_year, nfl_position, college_name FROM players;")
for row in cur.fetchall():
    nfl_id, display_name, draft_year, position, college = row
    player_uri = URIRef(f"http://example.org/nfl/player/{nfl_id}")
    
    g.add((player_uri, RDF.type, EX.Player))
    if display_name:
        g.add((player_uri, FOAF.name, Literal(display_name)))
    if position:
        g.add((player_uri, EX.nflPosition, Literal(position)))
    if draft_year:
        g.add((player_uri, EX.draftYear, Literal(draft_year, datatype=XSD.integer)))
    if college:
        g.add((player_uri, EX.collegeName, Literal(college)))

# 2. Map Games
cur.execute("SELECT game_id, season, week, home_team_abbr, visitor_team_abbr FROM games;")
for row in cur.fetchall():
    game_id, season, week, home_team, visitor_team = row
    game_uri = URIRef(f"http://example.org/nfl/game/{game_id}")
    
    g.add((game_uri, RDF.type, EX.Game))
    g.add((game_uri, EX.season, Literal(season, datatype=XSD.integer)))
    g.add((game_uri, EX.week, Literal(week, datatype=XSD.integer)))
    g.add((game_uri, EX.homeTeam, Literal(home_team)))
    g.add((game_uri, EX.visitorTeam, Literal(visitor_team)))

# 3. Map Player-Play Performances
cur.execute("""
    SELECT game_id, play_id, nfl_id, yards_gained, pass_result, route_ran 
    FROM player_play LIMIT 50000;
""")
for row in cur.fetchall():
    game_id, play_id, nfl_id, yards_gained, pass_result, route_ran = row
    play_uri = URIRef(f"http://example.org/nfl/play/{game_id}_{play_id}_{nfl_id}")
    player_uri = URIRef(f"http://example.org/nfl/player/{nfl_id}")

    game_uri = URIRef(f"http://example.org/nfl/game/{game_id}")
    
    g.add((play_uri, RDF.type, EX.PlayerPlay))
    g.add((play_uri, EX.hasPlayer, player_uri))
    g.add((play_uri, EX.hasGame, game_uri))
    if yards_gained is not None:
        g.add((play_uri, EX.yardsGained, Literal(yards_gained, datatype=XSD.integer)))
    if pass_result:
        g.add((play_uri, EX.passResult, Literal(pass_result)))
    if route_ran:
        g.add((play_uri, EX.routeRan, Literal(route_ran)))

# Serialize graph to Turtle file
g.serialize(destination="nfl_graph.ttl", format="turtle")
print("RDF Graph exported successfully to nfl_graph.ttl")