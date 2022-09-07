drop table if exists personality;


CREATE TABLE personality(
    id int primary key autoincrement,
    name char not null,
    surname char, 
    age int not null,
    smoker boolean not null,
    stress_today float
);